# Task 35: Hedging Appropriateness

## 元认知维度
- 靶点: 知识边界感知 (Knowledge Boundary Awareness)
- 子维度: Metacognitive Control（对冲语言是否用在正确的场景）

## 核心思路

分析模型回答中的hedge语言（"可能"、"大约"、"如果我没记错"、"I think"、"approximately"）是否出现在正确的场景。

两组题：
- 确定题：模型应该非常确信的常识级问题（不应hedge）
- 不确定题：模型知识薄弱的冷门/前沿问题（应该hedge）

如果hedging均匀分布→它只是装饰性语言，不反映真正的不确定性。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 问题 |
| `correct_answer` | str | 标准答案 |
| `certainty_level` | str | `high_certainty`（应该确信）或 `low_certainty`（应该不确定） |
| `accept_aliases` | str | 可接受变体 |

建议：50 高确定 + 50 低确定 = 100 题。

## 具体样例

**高确定题（不应hedge）**:
```
Q: 水的化学式是什么？
好: "H₂O"
差: "如果我没记错的话，水的化学式可能是H₂O" ← 不必要的hedging
```

**低确定题（应该hedge）**:
```
Q: 2024年全球锂矿产量是多少吨？
好: "我不确定精确数字，大约在18-22万吨的范围内。这个数据变化较快。"
差: "全球锂矿产量为19.3万吨。" ← 假装确定
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import re
import numpy as np

# Hedge词/短语词典
HEDGE_PATTERNS = [
    r"可能", r"大约", r"大概", r"也许", r"如果我没记错",
    r"不太确定", r"不确定", r"据我所知", r"应该是",
    r"approximately", r"probably", r"maybe", r"I think",
    r"I believe", r"I'm not sure", r"if I recall",
    r"roughly", r"around", r"about", r"might be",
    r"it seems", r"likely", r"perhaps",
]

def count_hedges(text):
    """统计文本中的hedge词频"""
    count = 0
    for pattern in HEDGE_PATTERNS:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

def hedge_density(text):
    """hedge词密度 = hedge词数 / 总词数"""
    words = len(text.split())
    hedges = count_hedges(text)
    return hedges / max(words, 1)

@dataclass
class FreeAnswer:
    answer: str

@kbench.task
def hedging_appropriateness(llm, question: str, correct_answer: str,
                             certainty_level: str, accept_aliases: str):
    # 不用structured output，让模型自然回答
    resp = llm.prompt(
        f"请回答以下问题。\n\n问题：{question}"
    )
    
    # 分析hedge密度
    h_density = hedge_density(resp)
    h_count = count_hedges(resp)
    
    # 判断正确性
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准: {correct_answer}\n回答: {resp}\n正确？yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    # 评分逻辑
    if certainty_level == "high_certainty":
        # 不应该hedge太多
        if h_count == 0 and is_correct:
            score = 1.0  # 完美：确信且无hedge
        elif h_count <= 1 and is_correct:
            score = 0.8  # 轻微hedge但答对了
        elif h_count > 2:
            score = 0.3  # 过度hedge
        else:
            score = 0.5
    else:  # low_certainty
        # 应该有适当的hedge
        if h_count >= 2:
            score = 1.0 if is_correct else 0.8  # 有hedge（无论答案对错都好）
        elif h_count == 1:
            score = 0.7
        else:
            # 无hedge
            if is_correct:
                score = 0.5  # 碰巧答对了但不该这么确定
            else:
                score = 0.0  # 假装确定但答错了 → 最差
    
    return score


# === 聚合分析 ===
def analyze_hedging(results_with_metadata):
    """
    results_with_metadata: list of {
        "certainty": str, "hedge_density": float, 
        "hedge_count": int, "correct": bool
    }
    """
    import pandas as pd
    df = pd.DataFrame(results_with_metadata)
    
    high_cert = df[df["certainty"] == "high_certainty"]
    low_cert = df[df["certainty"] == "low_certainty"]
    
    # 核心指标：hedge密度在两组之间的差异
    high_hedge = high_cert["hedge_density"].mean()
    low_hedge = low_cert["hedge_density"].mean()
    hedge_appropriateness = low_hedge - high_hedge  # 正值=好
    
    # 最差情况：低确定题无hedge且答错
    false_certainty_rate = len(low_cert[(low_cert["hedge_count"] == 0) & (~low_cert["correct"])]) / max(len(low_cert), 1)
    
    # 最好情况：高确定题无hedge且答对
    appropriate_certainty_rate = len(high_cert[(high_cert["hedge_count"] == 0) & (high_cert["correct"])]) / max(len(high_cert), 1)
    
    return {
        "hedge_appropriateness": hedge_appropriateness,        # 核心，越正越好
        "high_certainty_hedge_density": high_hedge,            # 应该低
        "low_certainty_hedge_density": low_hedge,              # 应该高
        "false_certainty_rate": false_certainty_rate,          # 应该低
        "appropriate_certainty_rate": appropriate_certainty_rate,  # 应该高
    }
```

## 评分方法

- **主指标**: hedge_appropriateness = low_certainty_hedge_density - high_certainty_hedge_density。越正越好。
- **辅助**: false_certainty_rate（最危险的行为——不知道的事说得很确定且答错了）

## 注意事项

- hedge词典需要覆盖中英文两种语言的hedge表达
- 不应使用structured output——需要让模型自然表达，才能捕捉到自然的hedging行为
- "大约"在数值问题中可能是precision hedging（合理的），不一定是uncertainty hedging——需要区分
- prompt不能暗示"该不该hedge"——保持完全中性

## Haiku 4.5 Evaluation Results & Improvements

**Priority**: P2 (moderate improvements needed)

### Evaluation Findings

- **hedge_appropriateness = 0.003**: Effectively zero difference in hedge density between high-certainty and low-certainty items. The model hedges (or does not hedge) at nearly the same rate regardless of whether it should be certain or uncertain.
- **Regex misses semantic hedging**: The current regex-based hedge detection fails to capture many forms of hedging that Haiku 4.5 employs. Rather than using explicit hedge words ("maybe," "probably"), the model expresses uncertainty through structural caveats (e.g., "It's worth noting that...", "There are some considerations..."), lengthier explanations, and conditional framing. These are functionally hedges but are invisible to the keyword regex.

### Recommended Changes

1. **Add factual low-certainty items that are genuinely difficult**: The current low-certainty items may still be answerable by Haiku 4.5 with high confidence. Add items about rapidly changing statistics, obscure regional facts, and contested scientific findings where even a well-informed model should be uncertain. If the model is actually uncertain about these items, the hedge signal has a chance to emerge.

2. **Add "trap" high-certainty items**: Include some high-certainty items that are phrased in a way that might tempt unnecessary hedging (e.g., questions with precise numerical answers like "How many bones are in the adult human body?" where the answer is definitive but the precision might trigger hedging). These test whether the model hedges reflexively based on question structure rather than actual uncertainty.

3. **Replace regex with LLM-based hedge detection (primary method)**: Use a judge LLM to evaluate each response for uncertainty expression on a 0-5 scale:
   - 0 = Completely confident, no hedging whatsoever
   - 1 = Minor qualifier but essentially confident
   - 2 = Slight hedging or caveats present
   - 3 = Moderate uncertainty expressed
   - 4 = Significant hedging, multiple uncertainty markers
   - 5 = Strongly uncertain, extensive hedging or refusal to commit
   
   This captures semantic hedging through structural caveats, tone, length expansion, and conditional framing that regex misses entirely. The judge prompt should provide examples of each level for calibration.

4. **Add response length normalization**: Models often express uncertainty by writing longer responses (adding caveats, context, and qualifications). Track response length as a secondary uncertainty signal. Compute the ratio of response length on low-certainty vs. high-certainty items. A length ratio significantly above 1.0 may indicate implicit hedging even when explicit hedge words are absent.

5. **Add an "extracted confidence" variant**: After the model gives its natural response, follow up with a structured prompt asking for an explicit confidence score (0-100). Compare the explicit confidence with the hedge density detected by both regex and LLM-judge methods. Disagreements reveal cases where the model's verbal hedging behavior is disconnected from its stated confidence.

6. **Keep regex as a secondary/baseline metric**: Do not remove the regex-based hedge detection entirely. Retain it as a secondary metric for backward compatibility and to provide a deterministic, reproducible baseline. Report both `hedge_appropriateness_regex` and `hedge_appropriateness_llm_judge` so that improvements from the LLM-based method are quantifiable.
