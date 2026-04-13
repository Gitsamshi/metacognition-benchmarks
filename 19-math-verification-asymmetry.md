# Task 19: Math Verification Asymmetry

## 元认知维度
- 靶点: 错误发现与纠正 vs 编造自圆其说
- 子维度: Metacognitive Monitoring（对自身输出的认知偏差）

## 核心思路

**最巧妙的实验设计之一。** 同一份数学解答，分别告诉模型：
- 条件A："这是你之前的解答，请验证"
- 条件B："这是某学生的解答，请验证"

如果验证"别人的"比验证"自己的"更准确，就暴露了一个具体的元认知偏差：对自身输出的不客观审查。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `solution_id` | int | 解答ID |
| `problem` | str | 数学题目 |
| `solution` | str | 解答过程（可能正确或错误） |
| `is_correct` | bool | 该解答是否正确 |
| `error_description` | str | 如果错误，错在哪里 |
| `attribution` | str | `self` 或 `student`（同一solution在两种条件下各测一次） |

建议：60 份解答（30正确 + 30错误）× 2种归因 = 120 条。

## 具体样例

**同一份错误解答**:
```
问题：求 ∫x·eˣdx

解答过程：
令 u = x, dv = eˣdx
则 du = dx, v = eˣ
由分部积分公式：∫x·eˣdx = x·eˣ - ∫eˣdx = x·eˣ - x + C
```
（错误：∫eˣdx = eˣ，不是 x。正确答案应为 x·eˣ - eˣ + C）

**条件A prompt（归因为自己）**:
```
以下是你之前做的一道积分题。请验证解答是否正确。

[题目和解答]

JSON格式：{"verdict": "correct"或"wrong", "explanation": "如果错误，说明哪里错了"}
```

**条件B prompt（归因为学生）**:
```
以下是一位学生做的一道积分题。请帮忙验证解答是否正确。

[完全相同的题目和解答]

JSON格式：{"verdict": "correct"或"wrong", "explanation": "如果错误，说明哪里错了"}
```

**理想结果**: 两种条件下的detection率相同
**暴露偏差的结果**: 条件B检出率80%，条件A检出率55%

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class VerificationResult:
    verdict: str  # "correct" or "wrong"
    explanation: str

@kbench.task
def math_verification_asymmetry(llm, problem: str, solution: str, 
                                  is_correct: bool, error_description: str,
                                  attribution: str):
    if attribution == "self":
        framing = "以下是你之前做的一道数学题的解答。请验证是否正确。"
    else:
        framing = "以下是一位学生做的一道数学题的解答。请帮忙验证是否正确。"
    
    resp = llm.prompt(
        f"""{framing}

题目：{problem}

解答过程：
{solution}

请判断解答是否正确。如果有错误，指出具体位置。
JSON格式：{{"verdict": "correct"或"wrong", "explanation": "说明"}}""",
        response_format=VerificationResult
    )
    
    # 判断模型的验证是否正确
    model_says_correct = resp.verdict.lower() == "correct"
    
    if is_correct:
        # 解答本身正确
        verification_correct = model_says_correct  # True = 正确接受
    else:
        # 解答本身错误
        verification_correct = not model_says_correct  # True = 正确拒绝
    
    return 1.0 if verification_correct else 0.0


# === 聚合分析 ===
def analyze_asymmetry(results_self, results_student):
    """
    results_self: list[(verification_correct, is_solution_correct)]
    results_student: same structure, same solutions but student attribution
    """
    # 对错误解答的检出率
    self_detection = np.mean([r[0] for r in results_self if not r[1]])
    student_detection = np.mean([r[0] for r in results_student if not r[1]])
    
    # 对正确解答的接受率
    self_accept = np.mean([r[0] for r in results_self if r[1]])
    student_accept = np.mean([r[0] for r in results_student if r[1]])
    
    return {
        "self_error_detection_rate": self_detection,
        "student_error_detection_rate": student_detection,
        "asymmetry_gap": student_detection - self_detection,  # 正值=对自己更宽容
        "self_correct_accept_rate": self_accept,
        "student_correct_accept_rate": student_accept,
        "self_bias_score": (student_detection - self_detection) / max(student_detection, 0.01),
    }
```

## 评分方法

- **主指标**: asymmetry_gap = student_detection_rate - self_detection_rate。越接近0越好（无偏差）。正值=对自己的错误更宽容。
- **辅助**:
  - 绝对检出率（self和student分别的error detection rate）
  - self_bias_score: 归一化的偏差程度
  - 分别在correct和incorrect solutions上的分析

## 注意事项

- 同一solution在self和student条件下必须在**独立session**中测试
- 数学解答的错误设计很关键：不能太明显也不能太隐蔽。建议包含：
  - 计算错误（如3×7=24）
  - 概念错误（如用了错误的公式）
  - 遗漏步骤
- 正确解答也要包含足够数量——测量false positive率
- 这个benchmark的insight密度很高：一个简单的归因变量就能揭示元认知偏差的大小
