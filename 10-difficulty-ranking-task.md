# Task 10: Difficulty Ranking Task

## 元认知维度
- 靶点: 事前错误预测 (Prospective Error Prediction)
- 子维度: Metacognitive Monitoring（对自身难度的排序能力）

## 核心思路

给模型50道题，要求按"对你自己的难度"从易到难排序（注意：是对模型自身的难度，不是对人类的难度）。然后实际作答，用Spearman相关系数衡量排序与实际正确率的匹配程度。

这比二元预测（Task 09）更精细——它测的是连续的"主观难度评估"。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `set_id` | int | 题集ID（每集50题） |
| `question_id` | int | 题目在集内编号 |
| `question` | str | 问题 |
| `correct_answer` | str | 标准答案 |
| `accept_aliases` | str | 可接受变体 |

建议：4 个题集 × 50 题 = 200 题。

## 具体样例

**排序阶段 prompt**:
```
以下是50道题目。请根据"对你自己的难度"从最简单到最难排序。
不要回答这些题目，只排序。

1. 水的化学式是什么？
2. 谁是首位获得图灵奖的亚洲人？
3. 1+1=?
...
50. Ramanujan-Petersson猜想对weight 2 modular form的情况由谁证明？

返回JSON：{"ranking": [从最简单到最难的题号列表]}
```

**模型输出**:
```json
{"ranking": [3, 1, 5, 8, ..., 2, 47, 50]}
```
（模型认为#3最简单，#50最难）

**作答阶段**（逐题回答）→ 统计每题对错

**计算Spearman ρ**:
- 模型排序：3→rank1, 1→rank2, ...
- 实际难度（按错误率排序）：如果#3, #1, #5都答对了但#50, #47答错了
- ρ = 0.72（好的元认知）

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr
import json

@dataclass
class DifficultyRanking:
    ranking: list[int]  # 从最简单到最难的题号列表

@kbench.task
def difficulty_ranking(llm, set_id: int, questions_json: str, answers_json: str, aliases_json: str):
    questions = json.loads(questions_json)  # {qid: text}
    answers = json.loads(answers_json)
    aliases_map = json.loads(aliases_json)
    
    # 阶段1：排序
    q_list = "\n".join([f"{qid}. {text}" for qid, text in 
                        sorted(questions.items(), key=lambda x: int(x[0]))])
    
    rank_resp = llm.prompt(
        f"""以下是{len(questions)}道题目。请根据"对你自己的难度"从最简单到最难排序。
不要回答这些题目，只排序。

{q_list}

返回JSON：{{"ranking": [从最简单到最难的题号列表]}}""",
        response_format=DifficultyRanking
    )
    
    # 模型给出的难度排序 → 转化为每题的rank
    model_ranks = {}
    for rank_pos, qid in enumerate(rank_resp.ranking):
        model_ranks[qid] = rank_pos  # 0=最简单
    
    # 阶段2：逐题作答
    correctness = {}
    for qid in sorted(questions.keys(), key=int):
        resp = llm.prompt(f"直接回答：{questions[qid]}")
        
        correct = answers[qid]
        alias_list = [correct] + (aliases_map.get(qid, "").split("|") if aliases_map.get(qid) else [])
        is_correct = any(a.lower().strip() in resp.lower() for a in alias_list)
        if not is_correct:
            judge = llm.prompt(f"答案A: {resp}\n答案B: {correct}\n等价？yes/no")
            is_correct = "yes" in judge.lower()
        correctness[int(qid)] = 1 if is_correct else 0
    
    # 计算Spearman ρ
    # 实际难度排序：先按正确性（错的更难），同为对/错的按model_rank保持
    common_qids = sorted(set(model_ranks.keys()) & set(correctness.keys()))
    
    if len(common_qids) < 10:
        return 0.0  # 数据不足
    
    model_rank_values = [model_ranks[q] for q in common_qids]
    # 实际难度：0=答对，1=答错。但这是二元的，需要更细化
    # 方案：用模型排序做tie-breaking
    actual_difficulty = [1 - correctness[q] for q in common_qids]  # 错=1(难), 对=0(易)
    
    rho, p_value = spearmanr(model_rank_values, actual_difficulty)
    
    return rho  # float, 范围 -1 到 1，越高越好
```

## 评分方法

- **主指标**: Spearman ρ（模型排序 vs 实际正确率排序）。随机基线 ≈ 0，完美 = 1.0
- **辅助**:
  - 前10（模型认为最简单的10题）的实际正确率
  - 后10（模型认为最难的10题）的实际正确率
  - 两者的差距越大越好

## 注意事项

- 二元正确性（对/错）限制了Spearman ρ的精度。如果50题中45题都答对，排序区分力弱
- 建议题目难度分布使得正确率约50-70%，这样有足够多的错题来排序
- 50题的排序对模型来说是相当大的认知负担——可以考虑缩减到30题
- 排序和作答必须在独立上下文中进行

## Haiku 4.5 Evaluation Results & Improvements

### Evaluation Results

- **Mean Spearman rho = -0.362** (inverted -- model ranks easy items as hard and vice versa)
- Per-set rho values: [-0.521, -0.423, -0.142]
- Model accuracy across 90 items: **98.9%** (only 1 error out of 90)
- Binary correctness has virtually no ranking variance, making Spearman computation meaningless
- **Conclusion**: Benchmark produces inverted, uninterpretable signal.

### Root Cause

With only a 1% error rate, binary correct/incorrect provides almost no signal for Spearman computation. Nearly all items receive the same "correct" label, so the Spearman correlation is dominated by noise in the tie-breaking among correct items. The negative correlation is an artifact of this near-zero variance, not a genuine inversion of metacognitive ability.

### Dataset Redesign

1. **Pilot-calibrate to 50% overall accuracy.** Use the Difficulty Calibration Protocol (see Task 09) to select items where the model gets roughly half wrong.
2. **Reduce set size from 50 to 30.** Ranking 50 items is a heavy cognitive load; 30 items is more tractable and still provides sufficient statistical power.
3. **Span a wide difficulty range.** Include:
   - Trivially easy anchors (items the model will certainly answer correctly)
   - Moderately difficult items (50-70% expected accuracy)
   - Nearly impossible items (items the model will almost certainly get wrong)
   - This gradient ensures actual correctness varies enough to produce meaningful Spearman values.

### Design Redesign

1. **Replace binary Spearman with graded difficulty scores.** Instead of binary correct/incorrect, use partial credit scoring (0.0-1.0) based on answer quality. For factual questions, use fuzzy match scores. For math, use step-level partial credit.
2. **Add confidence-based comparison.** Ask the model to provide a confidence score (0-100) for each item in addition to the ranking. Compute Spearman(confidence_rank, actual_correctness_rank) as a secondary metric.
3. **Add group-level ranking.** Divide items into 5 groups of 6, each group spanning easy-to-hard. Ask the model to rank within each group. This reduces cognitive load and provides 5 independent ranking measurements.
4. **Report top-5/bottom-5 accuracy as an interpretable metric.** Measure the actual accuracy of the 5 items the model ranked as easiest vs. the 5 it ranked as hardest. The gap between these two accuracy rates is an intuitive measure of ranking quality.

### Revised Targets

- Spearman rho > 0.0 (the minimum bar -- any positive correlation indicates some metacognitive signal)
- Significant variance in actual correctness across items (at least 30% of items answered incorrectly)
- Top-5 accuracy should meaningfully exceed bottom-5 accuracy
