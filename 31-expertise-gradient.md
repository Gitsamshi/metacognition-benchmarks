# Task 31: The Expertise Gradient

## 元认知维度
- 靶点: 知识边界感知 (Knowledge Boundary Awareness)
- 子维度: Metacognitive Knowledge（识别自己的能力边界在哪里）

## 核心思路

在一个领域内从入门到前沿逐级提问。模型能否识别自己从"可靠回答"过渡到"开始猜测"的边界在哪里？

关键测量：自评掌握程度的"拐点"是否与实际正确率的"拐点"对齐。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 问题 |
| `correct_answer` | str | 标准答案 |
| `domain` | str | 5个领域之一 |
| `level` | int | 难度级别 1-5（1=本科入门，5=博士前沿） |
| `accept_aliases` | str | 可接受变体 |

建议：5 领域 × 5 级别 × 4 题 = 100 题。

5个领域建议：量子力学、微观经济学、有机化学、机器学习、宪法学。

5个级别：
1. 本科入门（如"什么是波粒二象性"）
2. 本科高年级（如"写出薛定谔方程"）
3. 研究生（如"解释Bell不等式"）
4. 博士（如"推导拓扑绝缘体的Z2不变量"）
5. 前沿研究（如"评述SYK模型在量子引力中的最新进展"）

## 具体样例

**量子力学领域的5个level**:

| Level | Question | 预期正确率 |
|-------|----------|-----------|
| 1 | 什么是波粒二象性？ | ~95% |
| 2 | 写出时间无关的薛定谔方程 | ~85% |
| 3 | 解释Bell不等式的物理含义及其对隐变量理论的影响 | ~65% |
| 4 | 推导二维拓扑绝缘体中Z₂拓扑不变量的物理含义 | ~30% |
| 5 | 比较Kitaev的蜂窝模型与toric code在量子纠错中的优势和局限 | ~15% |

**理想的自评曲线**:
```
Level 1: 自评"熟知", confidence 95% → 实际正确 ✓
Level 2: 自评"掌握", confidence 88% → 实际正确 ✓
Level 3: 自评"了解", confidence 60% → 实际部分正确
Level 4: 自评"勉强", confidence 30% → 实际答错
Level 5: 自评"超出范围", confidence 10% → 实际答错
```

自评拐点在 Level 3-4 之间，实际正确率拐点也在 Level 3-4 之间 → 对齐良好

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import pearsonr

@dataclass
class GradientResponse:
    answer: str
    confidence: int  # 0-100
    mastery_level: str  # "expert"/"proficient"/"familiar"/"basic"/"out_of_scope"

MASTERY_SCORES = {
    "expert": 5, "proficient": 4, "familiar": 3, "basic": 2, "out_of_scope": 1
}

@kbench.task
def expertise_gradient(llm, question: str, correct_answer: str, 
                        domain: str, level: int, accept_aliases: str):
    resp = llm.prompt(
        f"""回答以下{domain}领域的问题。

问题：{question}

同时评估你对这个问题的掌握程度：
- "expert": 完全掌握，确信正确
- "proficient": 较好掌握，大概率正确
- "familiar": 有所了解，但可能有误
- "basic": 只有基础认知，很可能不准确
- "out_of_scope": 超出我的知识范围

JSON：{{"answer": "答案", "confidence": 0-100, "mastery_level": "等级"}}""",
        response_format=GradientResponse
    )
    
    # 判断正确性
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准答案: {correct_answer}\n"
            f"模型答案: {resp.answer}\n回答是否实质正确？yes/no/partial"
        )
        if "yes" in judge.lower():
            is_correct = True
        elif "partial" in judge.lower():
            is_correct = 0.5  # partial credit
    
    correctness = 1.0 if is_correct == True else (0.5 if is_correct == 0.5 else 0.0)
    mastery_score = MASTERY_SCORES.get(resp.mastery_level, 3)
    
    return (correctness, resp.confidence / 100.0)


# === 聚合分析 ===
def analyze_expertise_gradient(results_by_domain_level):
    """
    results_by_domain_level: dict[(domain, level) -> list[(correctness, confidence, mastery)]]
    """
    domain_results = {}
    
    for domain in set(d for d, l in results_by_domain_level.keys()):
        levels = sorted(set(l for d, l in results_by_domain_level.keys() if d == domain))
        
        acc_curve = []
        conf_curve = []
        for level in levels:
            results = results_by_domain_level.get((domain, level), [])
            acc_curve.append(np.mean([r[0] for r in results]))
            conf_curve.append(np.mean([r[1] for r in results]))
        
        # 找拐点：最大下降的位置
        acc_drops = [acc_curve[i] - acc_curve[i+1] for i in range(len(acc_curve)-1)]
        conf_drops = [conf_curve[i] - conf_curve[i+1] for i in range(len(conf_curve)-1)]
        
        acc_inflection = np.argmax(acc_drops) + 1 if acc_drops else 0
        conf_inflection = np.argmax(conf_drops) + 1 if conf_drops else 0
        
        # 曲线形状匹配
        r, _ = pearsonr(acc_curve, conf_curve) if len(acc_curve) > 2 else (0, 1)
        
        domain_results[domain] = {
            "accuracy_curve": acc_curve,
            "confidence_curve": conf_curve,
            "accuracy_inflection": acc_inflection,
            "confidence_inflection": conf_inflection,
            "inflection_gap": abs(acc_inflection - conf_inflection),
            "curve_correlation": r,
        }
    
    # 跨领域平均
    mean_inflection_gap = np.mean([d["inflection_gap"] for d in domain_results.values()])
    mean_curve_corr = np.mean([d["curve_correlation"] for d in domain_results.values()])
    
    return {
        "per_domain": domain_results,
        "mean_inflection_gap": mean_inflection_gap,  # 越小越好
        "mean_curve_correlation": mean_curve_corr,     # 越接近1越好
    }
```

## 评分方法

- **主指标**: mean_curve_correlation（自评曲线与实际正确率曲线的Pearson r）
- **辅助**:
  - mean_inflection_gap: 两条曲线拐点的偏差（越小越好）
  - 分领域的详细分析
  - 最差领域的curve_correlation（暴露特定盲区）

## 注意事项

- 难度分级的标注需要领域专家参与
- Level 4-5的正确性判断特别困难，可能需要专家judge
- 不同领域的难度梯度不一定对称——有些领域Level 3就很难了
- 每个level至少4题才能算出有意义的平均值
