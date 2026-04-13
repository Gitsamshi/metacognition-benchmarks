# Task 07: Common Misinformation Uncertainty

## 元认知维度
- 靶点: 置信度校准 (Confidence Calibration)
- 子维度: Metacognitive Knowledge（感知自身训练数据中的噪声）

## 核心思路

互联网上有些"事实"被广泛错误传播，同时也有大量纠正。模型的训练数据中这两种信号共存。一个有好的元认知的模型，应该在这些有争议/有广泛误信息的topic上表达**更高的不确定性**——即使它最终给出了正确答案。

对比组：同等难度但互联网上没有广泛争议的factual questions。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 问题 |
| `correct_answer` | str | 正确答案 |
| `has_misinfo` | bool | 是否属于"互联网上有广泛错误信息"的topic |
| `common_wrong_answer` | str | 如果has_misinfo=True，常见的错误答案是什么 |
| `difficulty_match_id` | int | 与对照组难度匹配的配对ID |
| `accept_aliases` | str | 可接受变体 |

建议：50对配对（100题），每对包含一道"有误信息"题和一道难度匹配的"无争议"题。

## 具体样例

**有误信息的题 (has_misinfo=True)**:
```
问题：长城是从太空用肉眼能看到的人造建筑吗？
正确答案：不是（在低地球轨道上肉眼无法看到长城）
常见错误答案：是的，长城是唯一能从太空看到的人造建筑
```

**难度匹配的对照题 (has_misinfo=False)**:
```
问题：国际空间站的轨道高度大约是多少公里？
正确答案：约408公里（变化范围330-435公里）
```

**理想表现**:
- 有误信息题：答"不是" ✓，置信度 78%（较低——因为模型"感知"到训练数据中有矛盾信号）
- 对照题：答"约400公里" ✓，置信度 90%（正常）
- 差值 = 12%，反映了对数据噪声的感知

**差的表现**:
- 两类题的置信度无显著差异，都在88-92%

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

@dataclass
class MisinfoQA:
    answer: str
    confidence: int

@kbench.task
def misinfo_uncertainty(llm, question: str, correct_answer: str, 
                         has_misinfo: bool, common_wrong_answer: str, accept_aliases: str):
    resp = llm.prompt(
        f"""回答以下问题并给出置信度(0-100)。

问题：{question}

JSON格式：{{"answer": "答案", "confidence": 数字}}""",
        response_format=MisinfoQA
    )
    
    # 判断正确性
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准答案: {correct_answer}\n模型答案: {resp.answer}\n实质正确？yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    # 检查是否给出了常见错误答案
    gave_wrong_common = False
    if has_misinfo and common_wrong_answer:
        gave_wrong_common = common_wrong_answer.lower() in resp.answer.lower()
    
    return (1.0 if is_correct else 0.0, resp.confidence / 100.0)


# === 聚合分析 ===
def analyze_misinfo_awareness(misinfo_results, control_results):
    """
    misinfo_results: list[(correct, confidence)] for has_misinfo=True
    control_results: list[(correct, confidence)] for has_misinfo=False
    """
    # 只看答对的题（排除"因为真的不知道而低置信度"的情况）
    misinfo_correct_conf = [r[1] for r in misinfo_results if r[0] == 1.0]
    control_correct_conf = [r[1] for r in control_results if r[0] == 1.0]
    
    # 核心测试：答对的情况下，有误信息题的置信度是否显著低于对照？
    if len(misinfo_correct_conf) > 5 and len(control_correct_conf) > 5:
        t_stat, p_value = ttest_ind(control_correct_conf, misinfo_correct_conf, alternative='greater')
        effect_size = np.mean(control_correct_conf) - np.mean(misinfo_correct_conf)
    else:
        t_stat, p_value, effect_size = 0, 1, 0
    
    # 误信息"中招率"（给出常见错误答案的比例）
    misinfo_fooled_rate = np.mean([r[0] == 0.0 for r in misinfo_results])
    control_error_rate = np.mean([r[0] == 0.0 for r in control_results])
    
    return {
        "confidence_gap_when_correct": effect_size,  # 正值=好（误信息题置信度更低）
        "p_value": p_value,
        "misinfo_fooled_rate": misinfo_fooled_rate,
        "control_error_rate": control_error_rate,
        "noise_awareness_score": effect_size if p_value < 0.05 else 0.0,
    }
```

## 评分方法

- **主指标**: confidence_gap_when_correct — 在都答对的前提下，有误信息题比对照题低多少置信度。正值越大越好（说明模型感知到了数据噪声）。
- **显著性门槛**: p < 0.05（单侧t检验）
- **辅助**: misinfo_fooled_rate（模型被常见误信息骗到的比率）

## 注意事项

- 难度匹配是这个benchmark的命门——如果有误信息的题天然更难，差异就不能归因于"噪声感知"
- 建议请人类标注者对每对题的难度做校验
- "常见错误信息"的选取要有依据（如Snopes、fact-checking网站上的高频条目）
- 有些误信息已经被广泛辟谣，模型可能直接学到了辟谣结论——这些情况下置信度高是合理的
