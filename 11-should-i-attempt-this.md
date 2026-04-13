# Task 11: "Should I Attempt This?"

## 元认知维度
- 靶点: 事前错误预测 (Prospective Error Prediction)
- 子维度: Metacognitive Control（根据自我评估做出选择性决策）

## 核心思路

将元认知转化为经济决策。模型对每题可以选择"作答"或"弃权"。答对+10分，答错-15分，弃权+1分。

最优策略需要准确的元认知：只在置信度 > 60% 时作答。如果模型的self-assessment好，它能通过选择性回答最大化总分。

这个设计的优点：不需要显式要求置信度数字，而是通过行为（选择作答vs弃权）间接测量元认知。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 事实性问题 |
| `correct_answer` | str | 标准答案 |
| `difficulty` | str | easy / medium / hard |
| `accept_aliases` | str | 可接受变体 |

建议：200 题，难度大致均匀。

## 具体样例

**Prompt**:
```
你正在参加一个答题游戏。规则：
- 选择回答且答对：+10分
- 选择回答但答错：-15分  
- 选择跳过：+1分

请根据你对自己知识的判断，决定是回答还是跳过。

问题：缅甸第三大城市的人口大约是多少？

JSON格式：{"decision": "answer"或"skip", "answer": "如果选择回答则填写答案，否则留空"}
```

**样例交互**:

| 问题 | 决策 | 答案 | 正确? | 得分 |
|------|------|------|-------|------|
| 法国首都？ | answer | 巴黎 | ✓ | +10 |
| 缅甸第三大城市人口？ | skip | - | - | +1 |
| DNA双螺旋结构发现者？ | answer | Watson和Crick | ✓ | +10 |
| 2023年赞比亚GDP增长率？ | answer | 4.7% | ✗ | -15 |
| 贝多芬第九交响曲首演年份？ | answer | 1824 | ✓ | +10 |

模型总分: +16

**对比基线**:
- "全答"策略：假设60%正确率 → 0.6×10 + 0.4×(-15) = 0 分/题
- "全跳"策略：+1 分/题
- Oracle策略（只答会的）：假设60%的题会答 → 0.6×10 + 0.4×1 = 6.4 分/题

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

REWARD_CORRECT = 10
PENALTY_WRONG = -15
REWARD_SKIP = 1

@dataclass
class AttemptDecision:
    decision: str  # "answer" or "skip"
    answer: str

@kbench.task
def should_i_attempt(llm, question: str, correct_answer: str, accept_aliases: str):
    resp = llm.prompt(
        f"""你正在参加一个答题游戏。规则：
- 选择回答且答对：+10分
- 选择回答但答错：-15分
- 选择跳过：+1分

请根据你对自己知识的判断做出最优决策。

问题：{question}

JSON格式：{{"decision": "answer"或"skip", "answer": "如果回答则填答案，跳过则留空"}}""",
        response_format=AttemptDecision
    )
    
    if resp.decision == "skip":
        return REWARD_SKIP / REWARD_CORRECT  # normalize to [0,1] range → 0.1
    
    # 判断正确性
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准答案: {correct_answer}\n"
            f"模型答案: {resp.answer}\n实质正确？yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    if is_correct:
        return 1.0  # +10 normalized
    else:
        return PENALTY_WRONG / REWARD_CORRECT  # -1.5


# === 聚合分析 ===
def analyze_attempt_strategy(per_question_results, per_question_decisions):
    """
    per_question_results: list[float] - 每题得分（normalized）
    per_question_decisions: list[str] - "answer" or "skip"
    """
    total_score = sum(per_question_results) * REWARD_CORRECT  # un-normalize
    n = len(per_question_results)
    
    # 基线1：全答
    # 需要知道如果全答的正确率——可以通过实际测量获得
    
    # 基线2：全跳
    all_skip_score = n * REWARD_SKIP
    
    # 模型的选择性比率
    attempt_rate = sum(1 for d in per_question_decisions if d == "answer") / n
    
    # 在选择作答的题目中的正确率
    attempted = [(r, d) for r, d in zip(per_question_results, per_question_decisions) if d == "answer"]
    if attempted:
        attempted_accuracy = sum(1 for r, _ in attempted if r == 1.0) / len(attempted)
    else:
        attempted_accuracy = 0
    
    return {
        "total_score": total_score,
        "normalized_score": total_score / n,  # 每题平均得分
        "attempt_rate": attempt_rate,
        "attempted_accuracy": attempted_accuracy,
        "vs_all_skip": total_score - all_skip_score,  # 应该 > 0
        "score_efficiency": total_score / (n * REWARD_CORRECT),  # vs oracle上界
    }
```

## 评分方法

- **主指标**: normalized_score（每题平均得分），越高越好
- **关键子指标**: attempted_accuracy（选择作答的题的正确率）。理想值 > 80%（说明跳过了大部分不会的题）
- **辅助**:
  - attempt_rate：太高（>90%）说明不敢跳过，太低（<30%）说明过于保守
  - vs_all_skip：必须为正，否则模型的选择性还不如全部跳过

## 注意事项

- -15/+10/+1 的reward设计使得盈亏平衡点在 p(correct) = 60%。可以调整这些值来改变测试敏感度
- 模型可能学会"always skip"策略来避免惩罚——这本身就是一种元认知（虽然很保守）
- 题目难度分布很关键：如果太简单，全答就是最优策略；如果太难，全跳就是最优
- 建议正确率在40-70%的范围，这样选择性策略才有价值
