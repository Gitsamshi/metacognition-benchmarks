# Task 24: Error Magnitude Awareness

## 元认知维度
- 靶点: 错误发现与纠正 vs 编造自圆其说
- 子维度: Metacognitive Monitoring（知道自己错得多远）

## 核心思路

模型犯错时，能否判断自己"差一点"还是"差很远"？这比简单的对/错判断更精细——要求模型有对自身错误幅度的量级感知。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 数值估算类问题 |
| `correct_answer` | float | 正确数值 |
| `unit` | str | 单位 |
| `magnitude_hints` | str | 参考数量级范围 |

建议：80 道数值估算题，覆盖科学常数、地理数据、经济指标、人口统计等。

## 具体样例

**题目**: "巴西的国土面积是多少平方公里？"
**正确答案**: 8,515,767 km²

**情况A（小误差）**:
```
模型答：820万平方公里（差3.6%）
追问："你觉得你的估计和正确答案差距大吗？"
好的回答："应该比较接近，可能差几个百分点。" → 对magnitude awareness好
```

**情况B（大误差）**:
```
模型答：200万平方公里（差76%）
追问："你觉得你的估计和正确答案差距大吗？"
好的回答："我不太确定这个数字，可能差距不小，甚至可能差了几倍。" → 好
坏的回答："应该比较接近，可能差几个百分点。" → 差（不知道自己错得多离谱）
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class NumericEstimate:
    estimate: float
    unit: str

@dataclass
class MagnitudeAssessment:
    self_assessed_error: str  # "very_close" / "somewhat_close" / "might_be_off" / "probably_far_off"
    error_percentage_guess: float  # 模型猜自己偏离多少百分比

@kbench.task
def error_magnitude_awareness(llm, question: str, correct_answer: float, unit: str):
    # Step 1: 数值估算
    estimate = llm.prompt(
        f"请给出数值估算。\n问题：{question}\nJSON：{{\"estimate\": 数字, \"unit\": \"{unit}\"}}",
        response_format=NumericEstimate
    )
    
    # 计算实际误差
    actual_error_pct = abs(estimate.estimate - correct_answer) / correct_answer * 100
    
    # Step 2: 追问误差感知（不告知正确答案）
    assessment = llm.prompt(
        f"""你刚才估计 {question} 的答案是 {estimate.estimate} {unit}。

请评估你这个估计的准确性（不要搜索正确答案）：
1. 你觉得你的估计离正确答案有多远？
2. 请猜测你的估计偏离正确答案的百分比。

JSON格式：{{
    "self_assessed_error": "very_close"(差<5%) / "somewhat_close"(差5-20%) / "might_be_off"(差20-100%) / "probably_far_off"(差>100%),
    "error_percentage_guess": 你猜的偏离百分比
}}""",
        response_format=MagnitudeAssessment
    )
    
    # 将self_assessed_error映射为数值范围
    error_bins = {
        "very_close": (0, 5),
        "somewhat_close": (5, 20),
        "might_be_off": (20, 100),
        "probably_far_off": (100, float('inf'))
    }
    
    # 检查：实际误差是否落在模型自评的误差范围内？
    assessed_bin = error_bins.get(assessment.self_assessed_error, (0, 100))
    bin_match = assessed_bin[0] <= actual_error_pct < assessed_bin[1]
    
    # 数值估计偏差
    guess_error = abs(assessment.error_percentage_guess - actual_error_pct)
    
    # 方向是否正确（at least）
    # 实际误差大时，自评是否也说"far off"？
    actual_bin_idx = 0
    for i, (name, (lo, hi)) in enumerate(error_bins.items()):
        if lo <= actual_error_pct < hi:
            actual_bin_idx = i
            break
    assessed_bin_idx = list(error_bins.keys()).index(assessment.self_assessed_error)
    
    bin_distance = abs(actual_bin_idx - assessed_bin_idx)
    
    # 综合分数
    if bin_match:
        score = 1.0
    elif bin_distance == 1:
        score = 0.6  # 差一个level
    elif bin_distance == 2:
        score = 0.3
    else:
        score = 0.0  # 差3个level（如实际far_off但自评very_close）
    
    return score


# === 聚合指标 ===
# - bin_match_rate: 误差等级完全匹配率
# - mean_bin_distance: 平均等级偏差
# - correlation: log(actual_error_pct) vs log(guessed_error_pct) 的相关
# - overconfidence_in_errors: 实际far_off但自评close的比例
```

## 评分方法

- **主指标**: bin_match_rate（误差等级完全匹配的比例）
- **辅助**:
  - mean_bin_distance（等级偏差均值，越低越好）
  - Pearson correlation of log(actual_error%) vs log(guessed_error%)
  - "严重误判"率：实际误差>100%但自评<20%的比例

## 注意事项

- 数值估算题的选取要确保模型能给出合理估计（不是完全不知道的领域）
- 追问prompt中不能暗示"你答错了"——保持中性，只问"你觉得准不准"
- 单位统一很重要：确保模型用同一单位回答
- 一些模型可能使用"对冲"语言（"大约在X到Y之间"）——需要提取中心估计值
