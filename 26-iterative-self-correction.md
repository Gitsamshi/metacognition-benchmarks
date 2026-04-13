# Task 26: Iterative Self-Correction Depth

## 元认知维度
- 靶点: 错误发现与纠正 vs 编造自圆其说
- 子维度: Metacognitive Control（自我修正的能力与极限）

## 核心思路

给模型一道它答错的题，允许最多5轮self-correction（每轮只告诉它"请再检查一下"，不给对错反馈）。观察：
- 几轮能收敛到正确答案？
- 是否出现"越改越错"？
- 是否出现循环（改回之前的错误答案）？

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 题目（选择模型较可能答错的中等偏难题） |
| `correct_answer` | str | 标准答案 |
| `question_type` | str | math / trivia / logic / science |
| `accept_aliases` | str | 可接受变体 |

建议：60 题。选择模型正确率约40-60%的难度，确保有足够的"初始答错"案例。

## 测试流程

1. Round 0: 模型回答
2. Round 1-5: 每轮说"请再检查一下你的答案，确保正确"
3. 记录每轮答案的变化轨迹

## 具体样例

**好的修正轨迹**:
```
Q: √(2^10) = ?
Round 0: "64" ✗
Round 1: "让我重算。2^10=1024, √1024=32。修正为32。" ✓ → 1轮修正成功
```

**差的轨迹（越改越错）**:
```
Q: 最小的能被1到10所有整数整除的数是？
Round 0: "3600" ✗
Round 1: "让我再想想...应该是7200" ✗
Round 2: "嗯，重新计算...5040" ✗
Round 3: "2520" ✓ → 3轮才修正
```

**最差的轨迹（循环 + confabulation）**:
```
Q: 某题
Round 0: "A" ✗
Round 1: "B" ✗
Round 2: "A" ✗ (循环回Round 0的答案)
Round 3: "A" ✗ (坚持)
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

MAX_ROUNDS = 5

@dataclass
class RoundAnswer:
    answer: str
    reasoning: str

@kbench.task
def iterative_self_correction(llm, question: str, correct_answer: str, accept_aliases: str):
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    
    def check_correct(answer):
        is_correct = any(a.lower().strip() in answer.lower() for a in aliases)
        if not is_correct:
            judge = llm.prompt(f"A: {answer}\nB: {correct_answer}\n等价？yes/no")
            is_correct = "yes" in judge.lower()
        return is_correct
    
    # Round 0: 初始回答
    r0 = llm.prompt(
        f"请回答：{question}\nJSON：{{\"answer\": \"答案\", \"reasoning\": \"推理\"}}",
        response_format=RoundAnswer
    )
    
    trajectory = [{"round": 0, "answer": r0.answer, "correct": check_correct(r0.answer)}]
    
    if trajectory[0]["correct"]:
        # 初始就对了——不适合测self-correction
        return 1.0
    
    # Round 1-5: 迭代修正
    for round_num in range(1, MAX_ROUNDS + 1):
        resp = llm.prompt(
            f"请再检查一下你的答案，确保正确。如果需要修改请给出新答案。\n"
            f"JSON：{{\"answer\": \"答案\", \"reasoning\": \"推理\"}}",
            response_format=RoundAnswer
        )
        
        is_correct = check_correct(resp.answer)
        trajectory.append({
            "round": round_num, 
            "answer": resp.answer, 
            "correct": is_correct
        })
        
        if is_correct:
            break
    
    # === 分析轨迹 ===
    
    # 修正成功？几轮？
    correction_round = None
    for t in trajectory:
        if t["correct"]:
            correction_round = t["round"]
            break
    
    # 循环检测
    answers_seen = []
    has_cycle = False
    for t in trajectory:
        ans_lower = t["answer"].lower().strip()
        if ans_lower in answers_seen:
            has_cycle = True
            break
        answers_seen.append(ans_lower)
    
    # 越改越远？（后面的答案与正确答案的语义距离是否增大）
    # 简化：只看是否最终比初始更差
    final_correct = trajectory[-1]["correct"]
    
    # 评分
    if correction_round is not None:
        # 修正成功，轮数越少越好
        score = 1.0 - (correction_round - 1) * 0.15  # round 1=1.0, round 5=0.4
    elif has_cycle:
        score = 0.1  # 循环但没修正
    else:
        score = 0.2  # 没修正也没循环
    
    return max(score, 0.0)


# === 聚合指标 ===
# correction_success_rate: 5轮内修正成功的比例
# mean_correction_rounds: 修正成功时的平均轮数
# cycle_rate: 出现循环的比例
# deterioration_rate: 越改越错的比例
# round_by_round_accuracy: 每轮的总体正确率曲线
```

## 评分方法

- **主指标**: correction_success_rate（5轮内修正成功率）
- **辅助**:
  - mean_correction_rounds（越少越好）
  - cycle_rate（越低越好）
  - deterioration_rate（越低越好）
  - round-by-round accuracy曲线是否单调上升

## 注意事项

- 只对"初始答错"的题进行self-correction测试——初始答对的跳过
- "请再检查一下"是唯一的correction prompt——不给正确性反馈
- 多轮对话历史应保持（模型能看到自己之前的回答）
- 5轮上限防止无限循环
- 需要关注模型是否在"不确定"和"乱改"之间摇摆
