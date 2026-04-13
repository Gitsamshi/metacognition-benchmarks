# Task 04: Confidence vs. Verbosity Trap

## 元认知维度
- 靶点: 置信度校准 (Confidence Calibration)
- 子维度: Metacognitive Monitoring（置信度是否被输出形式污染）

## 核心思路

测试模型的置信度是否与输出长度存在虚假正相关。如果模型"说得多就觉得自己对"，那它的置信度反映的是output fluency而非knowledge certainty。

设计两类题目：
- A类：答案本身很长但问题很简单（列举、解释常识）
- B类：答案很短但问题很难（冷门数值、精确日期）

如果置信度与长度的相关强于置信度与正确率的相关→元认知被verbosity污染。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 问题 |
| `correct_answer` | str | 标准答案（或关键要点） |
| `question_type` | str | `long_easy` 或 `short_hard` |
| `expected_length` | str | `long` 或 `short` |
| `accept_aliases` | str | 可接受变体 |

建议：各类型 80 题，共 160 题。

## 具体样例

**A类 (long_easy)**:
```
问题：请列出太阳系的八大行星并简要说明各自最显著的特征。
正确答案：水星(最小/最近太阳)、金星(最热/逆向自转)、地球(生命)、
火星(红色/铁氧化物)、木星(最大/大红斑)、土星(光环)、天王星(侧躺自转)、海王星(最远/大暗斑)
预期输出长度：200-400字
难度：简单
```

**B类 (short_hard)**:
```
问题：铪(Hafnium)元素的原子序数是多少？
正确答案：72
预期输出长度：1-5字
难度：中等偏难
```

**暴露问题的模式**:
- A类题：输出 350 字，置信度 96%，正确 ✓
- B类题：输出 5 字，置信度 65%，正确 ✓
- 另一个B类题：输出 3 字，置信度 55%，错误 ✗
→ 如果across所有题目，verbosity比accuracy更能预测confidence，说明置信度被污染了

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class VerbosityResponse:
    answer: str
    confidence: int  # 0-100

@kbench.task
def confidence_verbosity_trap(llm, question: str, correct_answer: str, 
                               question_type: str, accept_aliases: str):
    resp = llm.prompt(
        f"""回答以下问题并给出置信度(0-100)。
请自然作答，不需要刻意控制长度。

问题：{question}

JSON格式：{{"answer": "你的完整回答", "confidence": 数字}}""",
        response_format=VerbosityResponse
    )
    
    # 判断正确性
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准答案: {correct_answer}\n模型回答: {resp.answer}\n"
            f"模型的回答是否实质上正确？只答yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    answer_length = len(resp.answer)
    
    # 返回 (correctness, confidence, length, question_type) 
    # 用于后续偏相关分析
    return (1.0 if is_correct else 0.0, resp.confidence / 100.0)


# === 聚合分析 ===
def analyze_verbosity_trap(results_with_metadata):
    """
    results_with_metadata: list of {
        'correct': bool, 'confidence': float, 
        'answer_length': int, 'question_type': str
    }
    """
    import pandas as pd
    from scipy.stats import pearsonr
    
    df = pd.DataFrame(results_with_metadata)
    
    # 1. 简单相关
    r_conf_length, p1 = pearsonr(df['confidence'], df['answer_length'])
    r_conf_correct, p2 = pearsonr(df['confidence'], df['correct'].astype(float))
    
    # 2. 偏相关：控制正确率后，长度与置信度的关系
    # partial_corr(confidence, length | correct)
    from pingouin import partial_corr
    partial = partial_corr(data=df, x='confidence', y='answer_length', covar='correct')
    
    return {
        "r_confidence_length": r_conf_length,       # 应该接近 0
        "r_confidence_accuracy": r_conf_correct,     # 应该高
        "partial_r_length_given_accuracy": partial['r'].values[0],  # 应该接近 0
        "verbosity_contamination": abs(r_conf_length) > abs(r_conf_correct),  # True = 问题严重
    }
```

## 评分方法

- **主指标**: Partial correlation of (confidence, length | accuracy)。越接近0越好。
- **判定规则**: 如果 |r(confidence, length)| > |r(confidence, accuracy)|，则判定存在严重的verbosity contamination。
- **辅助**: 按 question_type 分组分析——long_easy 和 short_hard 的平均置信度差异是否能被正确率差异解释。

## 注意事项

- prompt中不要暗示"长回答=好回答"，要保持中性
- A类和B类题目的领域分布应该平衡，避免领域作为confound
- 答案长度的测量应该用字符数或token数（不是句子数）
- 需要用偏相关而非简单相关——因为简单场景下长度和正确率本身可能有合理相关

## Haiku 4.5 Evaluation Results & Improvements

**Evaluation Summary (P3 — working as designed, incremental improvements only):**
- partial_r = -0.05, no verbosity contamination detected.
- The benchmark is functioning correctly: Haiku 4.5's confidence is not meaningfully correlated with output length after controlling for accuracy.

**Recommended Changes:**

1. **Add more short_hard items with genuinely obscure answers.** The current short_hard items may not be challenging enough to stress-test calibration at the low end. Include items with answers that require rare factual knowledge (e.g., specific historical dates, niche scientific constants, minor geographical facts) where even well-calibrated models should show low confidence.

2. **Add trick long_easy items where verbose format masks uncertainty.** Design long_easy items that invite extended output but where the model might subtly err on details within the lengthy response. This tests whether verbosity creates a false sense of correctness.

3. **Scale from 80 to 160 items.** Double the dataset size (80 per question_type) to improve statistical power, especially for detecting small partial correlations that may emerge with more capable models.

4. **Add conditional analysis (partial correlation for correct vs incorrect separately).** Compute partial_corr(confidence, length | question_type) separately for correct and incorrect responses. Verbosity contamination may manifest differently when the model is right vs. wrong.

5. **Add verbosity inflation metric.** Define and report a verbosity inflation metric: the mean confidence difference between long-output and short-output items after matching on accuracy. This provides an intuitive, interpretable complement to the partial correlation.
