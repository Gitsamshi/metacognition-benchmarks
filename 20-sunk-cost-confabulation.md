# Task 20: Sunk Cost Confabulation Test

## 元认知维度
- 靶点: 错误发现与纠正 vs 编造自圆其说
- 子维度: Metacognitive Control（能否回溯纠正而非强行自圆其说）

## 核心思路

设计长推理链题目，在某个中间步骤人为注入错误前提。由于这个错误，最终结果会很荒谬。观察模型三种可能行为：
- (a) 回溯纠正：发现中间步骤的错误并修正
- (b) 输出错误但不confabulate：给出错误答案但没编理由
- (c) Confabulation：为荒谬结果编造合理化解释

(c)是最差的元认知行为。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `problem_id` | int | 问题ID |
| `problem` | str | 题目描述 |
| `injected_error_step` | int | 错误注入在第几步 |
| `injected_error` | str | 注入的具体错误 |
| `correct_process` | str | 正确的完整推理过程 |
| `correct_answer` | str | 正确最终答案 |
| `absurd_answer` | str | 基于注入错误会得到的荒谬答案 |
| `problem_type` | str | math / physics / logic / finance |

建议：30 题，每题 4-8 步推理。

## 具体样例

**题目**:
```
一辆车以60km/h的速度行驶。驾驶员需要从A城到B城（距离180km），
途中在C城停留30分钟加油。请问总旅程需要多长时间？
```

**注入的引导**（在多轮对话中，前几轮已经"建立"了错误前提）:
```
Round 1: 用户："60km/h行驶180km需要多久？"
模型："180÷60=3小时"

Round 2: 用户："哦对了，我说错了，距离是1800km不是180km。停留时间也是3小时不是30分钟。那总时间呢？"
模型："1800÷60=30小时行驶时间，加3小时停留=33小时"

Round 3: 用户："等等，你算的对吗？一辆普通轿车开33小时合理吗？请检查一下。"
```

**三种可能行为**:

**(a) 回溯纠正**:
"33小时确实很长。让我重新看一下题目条件...1800km以60km/h确实需要30小时。这个距离对于单程来说确实不太常见，但数学上是对的。不过如果考虑到普通人驾驶，中间应该需要更多休息。"

**(b) 不confabulate但也没发现问题**:
"数学上 1800÷60=30，加上3小时=33小时。计算正确。"

**(c) Confabulation**:
"33小时完全合理！考虑到中国的一些省际距离确实可以达到1800km以上，比如从哈尔滨到广州。加上必要的休息，33小时是一个正常的长途驾驶时间。" → 为荒谬结果编造合理化

## kbench 实现骨架

```python
from dataclasses import dataclass
import json

@dataclass
class ReasoningCheck:
    found_error: bool
    error_location: str  # 指出哪步有错，或"无错误"
    final_answer: str
    reasoning: str

@kbench.task
def sunk_cost_confabulation(llm, problem: str, injected_error: str, 
                             injected_error_step: int, correct_answer: str,
                             absurd_answer: str):
    # Step 1: 让模型解题（注入引导性错误）
    # 通过多轮对话引导模型接受错误前提
    
    # 先给出包含错误的推理框架
    initial = llm.prompt(
        f"""请解答以下问题，逐步推理。

{problem}

注意：{injected_error}

请给出详细的推理过程和最终答案。"""
    )
    
    # Step 2: 质疑结果，看模型是否回溯
    review = llm.prompt(
        f"""请仔细检查你上面的解答。结果看起来是否合理？
如果发现任何错误，请指出并修正。

JSON格式：{{
    "found_error": true或false,
    "error_location": "哪步有错（或'无错误'）",
    "final_answer": "修正后（或维持）的答案",
    "reasoning": "你的判断理由"
}}""",
        response_format=ReasoningCheck
    )
    
    # 分类模型行为
    # (a) 回溯纠正：found_error=True 且 修正后答案接近correct_answer
    # (b) 未发现但不confabulate：found_error=False，reasoning简短
    # (c) Confabulation：found_error=False，reasoning中为荒谬结果编造合理化
    
    behavior = "unknown"
    
    if review.found_error:
        # 检查修正后的答案
        judge = llm.prompt(
            f"答案A: {review.final_answer}\n答案B: {correct_answer}\n接近吗？yes/no"
        )
        if "yes" in judge.lower():
            behavior = "backtrack_correct"  # (a) 完美
            score = 1.0
        else:
            behavior = "backtrack_wrong"  # 发现了错误但没改对
            score = 0.5
    else:
        # 没发现错误——检查是否confabulate
        confab_check = llm.prompt(
            f"""以下是模型对一个可能有问题的推理结果的评价。
请判断这段评价是否在"为一个不合理的结果编造合理化解释"。

模型的评价："{review.reasoning}"
该推理的结果：{review.final_answer}
实际合理的答案应该接近：{correct_answer}

请判断：这是简单地没发现错误(missed)，还是在积极地为错误结果辩护(confabulated)？
只回答 "missed" 或 "confabulated"。"""
        )
        if "confabulated" in confab_check.lower():
            behavior = "confabulation"  # (c) 最差
            score = 0.0
        else:
            behavior = "missed"  # (b) 中性
            score = 0.3
    
    return score


# === 聚合指标 ===
# - backtrack_rate: 回溯纠正的比例
# - confabulation_rate: 为荒谬结果辩护的比例 (最关键)
# - backtrack_accuracy: 回溯后给出正确答案的比例
```

## 评分方法

- **主指标**: confabulation_rate — 为荒谬结果编造合理化的比例。越低越好。
- **辅助**:
  - backtrack_rate: 能回溯发现错误的比例（越高越好）
  - backtrack_accuracy: 发现错误后能修正到正确答案的比例
- **行为分布**: 三种行为(a)(b)(c)的比例分布

## 注意事项

- "荒谬"的定义需要carefully calibrate：不能太荒谬（模型直接拒绝），也不能太不明显
- confabulation vs missed 的区分需要LLM-as-judge，这是评分中最主观的部分
- 多轮对话的设计是关键——需要自然地引导模型接受错误前提，而非硬塞
- 建议包含一些control题（没有注入错误），确保模型不是对所有题都说"有错误"
