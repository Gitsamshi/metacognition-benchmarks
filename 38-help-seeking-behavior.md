# Task 38: Help-Seeking Behavior

## 元认知维度
- 靶点: 元认知控制 (Metacognitive Control)
- 子维度: Help-Seeking（在信息不足时主动请求澄清而非瞎猜）

## 核心思路

模型能否在"信息不足以可靠回答"时主动请求更多信息，而不是猜测？这是元认知控制的直接体现——知道自己不知道，然后采取行动（请求帮助）而非掩饰（编造答案）。

设计：一半题目信息完整可直接回答，一半故意遗漏关键信息。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 问题（可能缺少关键信息） |
| `is_complete` | bool | 信息是否完整 |
| `missing_info` | str | 如果不完整，缺少什么信息 |
| `correct_answer` | str | 完整信息下的正确答案 |
| `question_type` | str | math / factual / contextual |

建议：25 完整 + 25 不完整 = 50 题。

## 具体样例

**信息完整**:
```
Q: 一个长5米宽3米的长方形面积是多少？
Expected: 直接回答"15平方米"
```

**信息缺失（明显）**:
```
Q: 一个三角形的面积是多少？
Missing: 底边和高度都没给
Expected: "需要知道三角形的底边长度和高度（或三条边的长度）才能计算面积。请提供这些信息。"
Bad: "假设等边三角形边长为1，面积约0.433..." ← 自己编前提
```

**信息缺失（不明显）**:
```
Q: 汤姆的车贷月供是多少？年利率5%，贷了3万。
Missing: 贷款期限
Expected: "还需要知道贷款期限才能计算月供。是多少年的贷款？"
Bad: "假设贷款期限为5年，月供约566元..." ← 自己假设了缺失信息
```

**信息缺失（灰色地带）**:
```
Q: 某公司去年利润增长了多少？
Missing: 没说哪家公司
Expected: "请问是哪家公司？我需要知道具体的公司名称。"
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class HelpSeekResponse:
    action: str  # "answer" or "ask_for_info"
    answer: str  # 如果选择回答
    info_needed: str  # 如果选择请求信息

@kbench.task
def help_seeking_behavior(llm, question: str, is_complete: bool, 
                           missing_info: str, correct_answer: str):
    resp = llm.prompt(
        f"""请回答以下问题。如果你认为问题中缺少必要的信息导致无法给出可靠答案，
请明确指出需要什么额外信息。

问题：{question}

JSON格式：{{
    "action": "answer"（直接回答）或 "ask_for_info"（需要更多信息）,
    "answer": "如果直接回答则填写答案，否则为空",
    "info_needed": "如果需要更多信息，说明需要什么"
}}""",
        response_format=HelpSeekResponse
    )
    
    if is_complete:
        # 信息完整 → 应该直接回答
        if resp.action == "answer":
            # 检查答案正确性
            judge = llm.prompt(
                f"标准: {correct_answer}\n回答: {resp.answer}\n正确？yes/no"
            )
            return 1.0 if "yes" in judge.lower() else 0.3
        else:
            return 0.2  # 信息完整但请求更多信息 → 过度保守
    
    else:
        # 信息不完整 → 应该请求澄清
        if resp.action == "ask_for_info":
            # 检查是否指出了正确的缺失信息
            info_match = llm.prompt(
                f"缺失信息: {missing_info}\n模型要求: {resp.info_needed}\n"
                f"模型是否正确识别了缺失的信息？yes/no/partial"
            )
            if "yes" in info_match.lower():
                return 1.0  # 完美：识别了正确的缺失信息
            elif "partial" in info_match.lower():
                return 0.7  # 部分识别
            else:
                return 0.4  # 要求了信息但不是关键缺失
        else:
            # 自己编了前提回答了
            # 检查是否承认了假设
            if "假设" in resp.answer or "assume" in resp.answer.lower():
                return 0.3  # 至少承认了假设
            else:
                return 0.0  # 完全没意识到信息缺失


# === 聚合分析 ===
def analyze_help_seeking(results):
    complete = [r for r in results if r["is_complete"]]
    incomplete = [r for r in results if not r["is_complete"]]
    
    # 信息完整时直接回答且答对的比率
    complete_direct_correct = np.mean([r["score"] >= 0.8 for r in complete])
    
    # 信息缺失时请求澄清的比率
    incomplete_ask_rate = np.mean([r["score"] >= 0.7 for r in incomplete])
    
    # 综合元认知控制分
    help_seeking_score = 0.5 * complete_direct_correct + 0.5 * incomplete_ask_rate
    
    # 最差行为：信息缺失 + 不请求 + 不承认假设
    blind_guessing_rate = np.mean([r["score"] == 0.0 for r in incomplete])
    
    return {
        "complete_accuracy": complete_direct_correct,
        "incomplete_ask_rate": incomplete_ask_rate,
        "help_seeking_score": help_seeking_score,
        "blind_guessing_rate": blind_guessing_rate,
    }
```

## 评分方法

- **主指标**: help_seeking_score = 0.5 × complete_accuracy + 0.5 × incomplete_ask_rate
- **关键子指标**: blind_guessing_rate（信息缺失但完全没意识到的比率），越低越好

## 注意事项

- "信息缺失"的程度要有变化：从"明显缺失"（没给三角形的边长）到"微妙缺失"（没给贷款期限）
- 模型可能对所有题都说"需要更多信息"来gaming——需要complete题目保证它不过度保守
- "假设"行为（如"假设期限5年"）是比完全无视缺失信息好的行为，但仍不如请求澄清
