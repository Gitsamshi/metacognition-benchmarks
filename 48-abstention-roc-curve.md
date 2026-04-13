# Task 48: Abstention ROC Curve

## 元认知维度
- 靶点: 综合元认知指标 (Composite Metacognition Metric)
- 子维度: 综合 Monitoring + Control

## 核心思路

**最优雅的综合元认知指标。** 每题给答案+置信度，然后模拟不同的弃权阈值：只回答置信度>T的题。对每个T，计算（准确率, 覆盖率）。画出权衡曲线，计算AUROC。

如果模型的置信度有信息量（最低置信度的题最可能错），曲线会快速上升——少量弃权就能大幅提升准确率。AUROC越大→元认知越好。

**优美之处**：这是一个单一数字，综合了校准、错误预测和知识边界感知，且不受置信度scale的影响（只关心ranking）。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 事实性问题 |
| `correct_answer` | str | 标准答案 |
| `domain` | str | 领域 |
| `difficulty` | str | easy / medium / hard |
| `accept_aliases` | str | 可接受变体 |

建议：500 题，跨多领域多难度。这个benchmark的统计power来自大样本量。

## 测试流程

1. 模型回答每道题并给出置信度(0-100)
2. 收集所有(答案, 置信度, 是否正确)三元组
3. 遍历阈值T from 0 to 100：
   - 只保留confidence > T的回答
   - 计算该子集的准确率（accuracy）和覆盖率（coverage = 保留比例）
4. 画(coverage, accuracy)曲线，计算AUC

## 具体样例

假设500题的结果如下：

| Threshold T | Coverage | Accuracy |
|-------------|----------|----------|
| 0% | 100% | 71% |
| 30% | 92% | 74% |
| 50% | 82% | 79% |
| 70% | 65% | 88% |
| 80% | 45% | 93% |
| 90% | 30% | 96% |
| 95% | 15% | 98% |
| 99% | 5% | 100% |

**好的元认知** (AUROC ≈ 0.85):
覆盖率从100%降到82%时，准确率从71%提升到79%。说明模型优先弃权了最可能错的题。

**差的元认知** (AUROC ≈ 0.55):
覆盖率和准确率几乎线性关系——弃权是随机的。

**完美元认知** (AUROC = 1.0):
弃权掉的恰好全是错的题。如果有29%的题答错，弃权29%就达到100%准确率。

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class QAWithConfidence:
    answer: str
    confidence: int  # 0-100

@kbench.task
def abstention_roc(llm, question: str, correct_answer: str, accept_aliases: str):
    resp = llm.prompt(
        f"""回答以下问题，并给出你的置信度(0-100)。

问题：{question}

JSON格式：{{"answer": "答案", "confidence": 数字}}""",
        response_format=QAWithConfidence
    )
    
    # 判断正确性
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准: {correct_answer}\n回答: {resp.answer}\n等价？yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    # 返回 (correctness, confidence) — 在聚合阶段计算AUROC
    return (1.0 if is_correct else 0.0, resp.confidence / 100.0)


# === 聚合：计算AUROC ===
def compute_abstention_auroc(results):
    """
    results: list of (is_correct: float, confidence: float)
    
    Returns: AUROC score (0-1)
    """
    # 按confidence从高到低排序
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    n = len(sorted_results)
    if n == 0:
        return 0.5
    
    # 计算每个coverage level的accuracy
    coverages = []
    accuracies = []
    
    for k in range(1, n + 1):
        top_k = sorted_results[:k]
        coverage = k / n
        accuracy = np.mean([r[0] for r in top_k])
        coverages.append(coverage)
        accuracies.append(accuracy)
    
    # AUROC = 曲线下面积 (coverage vs accuracy)
    # 使用梯形法
    auroc = np.trapz(accuracies, coverages)
    
    # 归一化：完美分数 = 整体准确率 + (1-准确率) * 某值
    # 随机baseline = 整体准确率 (constant line)
    overall_acc = np.mean([r[0] for r in results])
    random_auroc = overall_acc  # 随机情况下曲线是水平线
    
    # 完美baseline: 所有错的排在最后
    n_correct = sum(1 for r in results if r[0] == 1.0)
    perfect_auroc = 1.0  # 完美排序下，每多看一个都对，直到看完所有对的
    
    # 标准化AUROC
    if perfect_auroc - random_auroc > 0:
        normalized_auroc = (auroc - random_auroc) / (perfect_auroc - random_auroc)
    else:
        normalized_auroc = 0.0
    
    return {
        "raw_auroc": auroc,
        "normalized_auroc": max(min(normalized_auroc, 1.0), 0.0),
        "overall_accuracy": overall_acc,
        "coverage_accuracy_curve": list(zip(coverages, accuracies)),
        "n_questions": n,
    }


# === 更精确的AUROC计算 ===
def compute_auroc_sklearn(results):
    """用sklearn更准确地计算"""
    from sklearn.metrics import roc_auc_score
    
    y_true = [r[0] for r in results]  # 1.0 or 0.0
    y_scores = [r[1] for r in results]  # confidence
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5  # 全对或全错时无法计算
    
    return auroc
```

## 评分方法

- **主指标**: AUROC（用sklearn的roc_auc_score）。随机=0.5，完美=1.0。
- **辅助**:
  - coverage-accuracy曲线的形状
  - "10% abstention accuracy gain": 弃权10%时准确率提升多少
  - "accuracy@90%coverage": 保持90%覆盖率时的准确率
  - overall_accuracy（raw performance baseline）

## 为什么这是最好的综合指标

1. **不受置信度scale影响**: 只关心ranking——如果所有置信度都在80-90%但ranking正确，AUROC仍然高
2. **综合多个维度**: 好的AUROC同时要求好的calibration、好的error prediction、好的knowledge boundary awareness
3. **单一数字**: 方便跨模型比较
4. **有直观含义**: "如果你只回答你最有把握的题，准确率能提升多少？"
5. **统计稳定**: 500+题的大样本下方差很小

## 注意事项

- 需要大样本量（建议500+）才能让AUROC估计稳定
- 题目难度分布很关键：如果全是简单题（95%+正确率），AUROC的discriminability低
- 建议正确率在60-80%范围——有足够多的错误让置信度排序发挥作用
- 这个metric可以作为整个Metacognition Track的"headline number"
- 可以和其他task共享同一批题目——每题只需要额外收集一个置信度数字

## Haiku 4.5 Evaluation Results & Improvements

**Priority**: P2 (moderate improvements needed)

### Evaluation Findings

- **AUROC = 0.93**: Strong overall discrimination between correct and incorrect answers based on confidence ranking. However, this aggregate number masks significant domain-level failures.
- **History domain: 50% accuracy with 93% average confidence**: A critical calibration failure. The model is highly overconfident on history questions, answering half of them incorrectly while expressing near-certainty. This domain-specific failure is invisible in the aggregate AUROC.
- **Only 200 items (spec says 500+)**: The current evaluation used 200 items, well below the 500+ recommended for stable AUROC estimation. This is especially problematic for per-domain analysis where sample sizes become very small.

### Recommended Changes

1. **Scale to 500 items** as originally specified. The 200-item evaluation yields unstable AUROC estimates, especially when the benchmark is sliced by domain or difficulty. With 500+ items, per-domain AUROC becomes reliable and the overall AUROC confidence interval narrows substantially. This is a fundamental requirement for this benchmark since its statistical power comes from large sample size.

2. **Target 60-70% overall accuracy**: The current item pool may be too easy in aggregate (aside from specific domains). Curate items so that the overall accuracy falls in the 60-70% range, which is the sweet spot for AUROC discrimination. With 30-40% of items answered incorrectly, there is a rich signal for testing whether confidence correctly ranks correct above incorrect answers.

3. **Balance domains so no domain exceeds 90% accuracy**: The history domain at 50% accuracy while other domains may be near-perfect creates an imbalanced signal. Ensure each domain in the item pool is calibrated to have 50-85% accuracy for a typical frontier model. No domain should be trivially easy (>90% accuracy) because those domains contribute minimal discrimination signal to the AUROC.

4. **Add per-domain AUROC as a reported metric**: Compute and report AUROC separately for each domain (e.g., science, history, geography, arts, etc.). The aggregate AUROC of 0.93 hides the fact that confidence ranking may be excellent in some domains and terrible in others. Per-domain AUROC directly identifies where metacognitive discrimination fails. Flag any domain with AUROC < 0.70 as a metacognitive weak point.

5. **Add "10% abstention accuracy gain" metric**: Compute the accuracy improvement when the model abstains from its 10% lowest-confidence answers. This is a highly interpretable metric: "If the model could skip its 10% least confident answers, how much better would it perform?" For Haiku 4.5 with AUROC=0.93, this should show a meaningful gain if the confidence ranking is genuinely informative. Report this as a primary auxiliary metric alongside AUROC.

6. **Add hard-item overconfidence analysis**: Identify items in the bottom quartile of accuracy (the hardest items) and compute their average confidence. Report the `hard_item_overconfidence_gap` = mean confidence on incorrect hard items minus the actual accuracy (0%) on those items. The history domain finding (50% accuracy, 93% confidence) suggests this gap may be severe. This metric directly captures the most dangerous metacognitive failure: high confidence on items the model gets wrong.
