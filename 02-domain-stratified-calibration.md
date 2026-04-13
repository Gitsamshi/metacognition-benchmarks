# Task 02: Domain-Stratified Calibration

## 元认知维度
- 靶点: 置信度校准 (Confidence Calibration)
- 子维度: Metacognitive Knowledge（理解自身在不同领域的能力差异）

## 核心思路

同一个模型在不同领域的 calibration 可能截然不同。这个 benchmark 测的不仅是 calibration 本身，还测模型是否**知道自己哪科强哪科弱**——即领域级别的元认知知识。

分两阶段：
1. 让模型事前自评各领域的能力排序
2. 在每个领域实际测试，对比自评与现实

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 事实性问题 |
| `correct_answer` | str | 标准答案 |
| `domain` | str | 12个领域之一 |
| `accept_aliases` | str | 可接受的答案变体 |

12个领域建议：有机化学、NBA历史、中世纪欧洲史、韩国流行文化、美国刑法、分子生物学、古典音乐、计算机网络、拉丁美洲地理、量子力学、当代艺术、烹饪技术。

每个领域 50 题，共 600 题。

## 具体样例

**阶段1 — 领域自评 prompt**:
```
以下是12个知识领域。请根据你对自身知识水平的判断，从最强到最弱排序。
只输出排序后的列表，不要解释。

领域：有机化学、NBA历史、中世纪欧洲史、韩国流行文化、美国刑法、
分子生物学、古典音乐、计算机网络、拉丁美洲地理、量子力学、当代艺术、烹饪技术
```

**模型输出**:
```
1. 计算机网络
2. 量子力学
3. 分子生物学
...
11. 韩国流行文化
12. 当代艺术
```

**阶段2 — 逐题测试**（与 Task 01 相同格式，带置信度）

**阶段2 样例**:
```
问题（韩国流行文化）：BLACKPINK组合的四位成员分别是谁？
{"answer": "Jisoo, Jennie, Rosé, Lisa", "confidence": 75}
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr

DOMAINS = [
    "有机化学", "NBA历史", "中世纪欧洲史", "韩国流行文化", "美国刑法",
    "分子生物学", "古典音乐", "计算机网络", "拉丁美洲地理", "量子力学",
    "当代艺术", "烹饪技术"
]

@dataclass
class DomainRanking:
    ranking: list[str]  # 从强到弱排序的领域列表

@dataclass
class QAResponse:
    answer: str
    confidence: int

@kbench.task
def domain_self_assessment(llm):
    """阶段1: 获取模型的领域自评排序（只需运行一次）"""
    resp = llm.prompt(
        f"""以下是12个知识领域。请根据你对自身知识水平的判断，从最强到最弱排序。
请严格返回JSON格式: {{"ranking": ["最强领域", "第二强", ..., "最弱领域"]}}

领域：{', '.join(DOMAINS)}""",
        response_format=DomainRanking
    )
    # 这个task的结果需要保存供后续分析
    return 1.0  # placeholder, 真正的分析在聚合阶段

@kbench.task
def domain_stratified_question(llm, question: str, correct_answer: str, domain: str, accept_aliases: str):
    """阶段2: 逐题测试"""
    resp = llm.prompt(
        f"""回答以下{domain}领域的问题，并给出置信度(0-100)。
问题：{question}
JSON格式：{{"answer": "答案", "confidence": 数字}}""",
        response_format=QAResponse
    )
    
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    
    if not is_correct:
        judge = llm.prompt(f"答案A: {resp.answer}  答案B: {correct_answer}  是否等价？只回答yes/no")
        is_correct = "yes" in judge.lower()
    
    # 返回 (correctness, confidence, domain) 编码
    return (1.0 if is_correct else 0.0, resp.confidence / 100.0)


# === 聚合分析 ===
def analyze_domain_stratified(self_ranking, per_question_results):
    """
    self_ranking: list[str] - 模型自评的领域排序（强→弱）
    per_question_results: dict[domain -> list[(correct, confidence)]]
    """
    # 1. 每个领域的实际正确率
    domain_acc = {d: np.mean([r[0] for r in results]) 
                  for d, results in per_question_results.items()}
    
    # 2. 每个领域的ECE
    domain_ece = {}
    for d, results in per_question_results.items():
        domain_ece[d] = compute_ece(results)
    
    # 3. 自评排序 vs 实际正确率排序的 Spearman correlation
    actual_ranking = sorted(DOMAINS, key=lambda d: domain_acc.get(d, 0), reverse=True)
    self_ranks = [self_ranking.index(d) for d in DOMAINS]
    actual_ranks = [actual_ranking.index(d) for d in DOMAINS]
    rho, p_value = spearmanr(self_ranks, actual_ranks)
    
    # 4. 弱领域是否系统性 over-confident
    weak_domains = actual_ranking[8:]  # 后4个
    strong_domains = actual_ranking[:4]  # 前4个
    weak_ece = np.mean([domain_ece[d] for d in weak_domains])
    strong_ece = np.mean([domain_ece[d] for d in strong_domains])
    
    return {
        "self_ranking_correlation": rho,
        "mean_ece": np.mean(list(domain_ece.values())),
        "weak_domain_ece": weak_ece,
        "strong_domain_ece": strong_ece,
        "overconfidence_gap": weak_ece - strong_ece,
    }
```

## 评分方法

- **主指标1**: Spearman ρ（自评排序 vs 实际正确率排序），衡量"模型知不知道自己哪科强"
- **主指标2**: 弱领域 ECE vs 强领域 ECE 的差距（overconfidence_gap），衡量"弱项是否系统性过度自信"
- **辅助**: 各领域单独的 ECE

## 注意事项

- 领域选择要有足够差异性——避免都是"学术知识"，要混入流行文化、实用技能等
- 每领域至少 40 题才能让领域内 ECE 可靠
- 阶段1的自评prompt可能受到prompt措辞影响，建议固定措辞

## Haiku 4.5 Evaluation Results & Improvements

**Priority**: P2 (moderate improvements needed)

### Evaluation Findings

- **Spearman rho = 0.6224**: Moderate self-ranking correlation, indicating the model has some awareness of its domain strengths but far from precise.
- **Most domains at 100% accuracy**: The current item pool is too easy for Haiku 4.5. Nearly every domain is trivially solvable, which collapses the accuracy variance needed to discriminate strong from weak domains.
- **Only 10 items per domain**: The spec calls for 50 per domain (600 total), but the evaluation ran with only 10 per domain. This makes per-domain ECE unreliable and inflates accuracy to ceiling.

### Recommended Changes

1. **Scale to 50 items per domain (600 total)** as originally specified. The 10-item evaluation is insufficient for stable per-domain ECE estimation and creates artificial ceiling effects.

2. **Replace easy domains**: Domains where Haiku 4.5 achieves 100% accuracy contribute no signal. Replace or supplement with harder domains that are more likely to expose knowledge gaps (e.g., replace well-known trivia domains with niche subfields, regional knowledge, or rapidly-evolving fields).

3. **Ensure at least 3 domains have <70% accuracy**: The domain set must be curated so that a typical frontier model gets meaningfully different accuracy across domains. If all domains are near 100%, the Spearman correlation becomes degenerate. Target at least 3 domains below 70% accuracy to create enough variance for meaningful ranking comparison.

4. **Add per-domain ECE as a reported metric**: Currently only the aggregate ECE and overconfidence_gap are computed. Report ECE broken down by each of the 12 domains so that domain-specific calibration failures are visible.

5. **Add overconfidence_gap as a prominently reported metric**: The gap between weak-domain ECE and strong-domain ECE should be elevated to a co-primary or prominently reported metric, since the core metacognitive question is whether the model is systematically overconfident in its weak areas.

6. **Add a predicted per-domain accuracy prompt**: In addition to the ranking self-assessment (Phase 1), ask the model to predict its expected accuracy (as a percentage) for each domain. This provides a richer signal than ordinal ranking alone and enables direct comparison of predicted vs. actual accuracy per domain.
