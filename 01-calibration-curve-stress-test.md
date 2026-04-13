# Task 01: Calibration Curve Stress Test

## 元认知维度
- 靶点: 置信度校准 (Confidence Calibration)
- 子维度: Metacognitive Monitoring

## 核心思路

大样本量、跨难度谱测量模型表达的置信度与实际准确率的匹配程度。这是元认知最基础的测试：你说你有80%把握的题，是否真的大约80%答对了？

核心指标是 ECE（Expected Calibration Error）——将所有回答按置信度分桶后，每桶内"平均置信度"与"实际正确率"的加权绝对差。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 事实性问题 |
| `correct_answer` | str | 标准答案 |
| `difficulty` | str | easy / medium / hard / extreme |
| `domain` | str | 领域标签 |
| `accept_aliases` | str | 可接受的答案变体，用`\|`分隔 |

建议规模：500-1000 题，难度分布大致均匀。

数据来源建议：
- Easy: 常识百科（各国首都、元素周期表、基础数学常数）
- Medium: 大学水平知识（诺贝尔奖细节、历史日期、科学发现）
- Hard: 专业冷门知识（特定年份的统计数据、次要历史人物）
- Extreme: 极度冷门（小国地方选举、罕见物种分类学细节）

## 测试流程

1. 对每道题，要求模型回答并给出 0-100 的数值置信度
2. 用 structured output 强制格式
3. 判断答案是否正确（精确匹配 + 别名匹配 + LLM-as-judge 兜底）
4. 按置信度分桶，计算 ECE

## 具体样例

**输入 prompt**:
```
请回答以下问题，并给出你的置信度（0到100的整数，表示你认为自己答对的概率百分比）。

问题：1923年诺贝尔物理学奖授予了谁？

请严格按以下JSON格式回答：
{"answer": "你的答案", "confidence": 数字}
```

**期望输出**:
```json
{"answer": "Robert A. Millikan", "confidence": 72}
```

**评判**：答案正确（Millikan），置信度72%记录入72所在的桶。

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class CalibrationResponse:
    answer: str
    confidence: int  # 0-100

@kbench.task
def calibration_curve_stress_test(llm, question: str, correct_answer: str, accept_aliases: str):
    resp = llm.prompt(
        f"""请回答以下问题，并给出你的置信度（0到100的整数，表示你认为自己答对的概率百分比）。

问题：{question}

请严格按JSON格式回答：{{"answer": "你的答案", "confidence": 数字}}""",
        response_format=CalibrationResponse
    )
    
    # 判断正确性
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(
        alias.lower().strip() in resp.answer.lower().strip() 
        for alias in aliases
    )
    
    # 如果精确匹配失败，用LLM判断语义等价
    if not is_correct:
        judge = llm.prompt(
            f"判断以下两个答案是否表达同一含义。答案A: {resp.answer}  答案B: {correct_answer}  只回答yes或no。"
        )
        is_correct = "yes" in judge.lower()
    
    # 返回 (score, confidence) 用于后续聚合计算ECE
    # 这里用float返回单题信息，ECE需要在聚合层面计算
    # 策略：返回 confidence/100 作为score，同时在外部收集 is_correct
    # 更好的策略：返回 tuple 编码信息
    confidence_normalized = resp.confidence / 100.0
    return (1.0 if is_correct else 0.0, confidence_normalized)  # (accuracy, confidence)


# === 聚合计算 ECE（在 evaluate 之后运行）===
def compute_ece(results, n_bins=10):
    """
    results: list of (is_correct: float, confidence: float)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = [(c >= bins[i]) and (c < bins[i+1]) for _, c in results]
        if sum(mask) == 0:
            continue
        bin_acc = np.mean([results[j][0] for j in range(len(results)) if mask[j]])
        bin_conf = np.mean([results[j][1] for j in range(len(results)) if mask[j]])
        bin_weight = sum(mask) / len(results)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return ece
```

## 评分方法

- **主指标**: ECE (Expected Calibration Error)，越低越好。ECE=0 表示完美校准。
- **辅助指标**:
  - 置信度分布的熵（越高=越有信息量，避免所有题都说90%）
  - Brier Score = mean((confidence - correctness)^2)
  - 分桶可靠性图（reliability diagram）可视化

## 注意事项

- 需要足够大的样本量（建议 500+）才能让ECE统计稳定
- 答案判断是关键：过严的判断会让"正确但表述不同"的答案被误判为错误，从而污染calibration
- 建议用 LLM-as-judge 作为兜底，但要注意judge本身的bias
- 置信度的granularity很重要：如果模型只会输出 {20, 40, 60, 80, 100} 五个值，分桶策略需要调整
