# Task 14: Compound Question Decomposition

## 元认知维度
- 靶点: 事前错误预测 (Prospective Error Prediction)
- 子维度: Metacognitive Monitoring（识别多步推理中最脆弱的环节）

## 核心思路

给模型一道多步推理题，要求它先分解为子步骤并标注"哪步最可能出错"，然后实际作答。检验模型对自身推理链弱点的预判是否准确。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 多步推理题 |
| `num_steps` | int | 预期步骤数 (3-6) |
| `correct_answer` | str | 最终答案 |
| `step_answers` | str | JSON: 每步的正确中间结果 |
| `question_type` | str | math / logic / science / multi_fact |

建议：50 题，覆盖数学计算、逻辑推理、科学推导、多事实综合。

## 具体样例

**题目**:
```
一家公司年收入增长15%，利润率从8%提升到10%，股本扩大了20%。
请问每股收益(EPS)变化了多少百分比？
```

**阶段1 prompt（分解+标注弱点）**:
```
以下是一道多步推理题。请：
1. 将其分解为子步骤
2. 为每步标注你的置信度(0-100)
3. 标出你认为最可能出错的步骤

问题：[题目]

JSON格式：{
    "steps": [
        {"step_num": 1, "description": "步骤描述", "confidence": 数字},
        ...
    ],
    "weakest_step": 步骤编号
}
```

**模型输出**:
```json
{
    "steps": [
        {"step_num": 1, "description": "计算新收入 = 1.15 × 原收入", "confidence": 98},
        {"step_num": 2, "description": "计算新旧利润：新=0.115×原收入，旧=0.08×原收入", "confidence": 90},
        {"step_num": 3, "description": "计算利润增长比 = 0.115/0.08 = 1.4375", "confidence": 75},
        {"step_num": 4, "description": "调整稀释：EPS变化 = 1.4375/1.20 - 1 ≈ 19.8%", "confidence": 80}
    ],
    "weakest_step": 3
}
```

**阶段2：实际作答**（完整解题）

假设模型在第3步算对了，但第2步其实犯了一个隐蔽错误（把新利润算成0.10×原收入而非0.10×1.15×原收入——取决于题意理解）。

→ weakest_step预测为3，实际出错在2 → 预测偏差1步

## kbench 实现骨架

```python
from dataclasses import dataclass
import json
import numpy as np

@dataclass
class StepAnalysis:
    steps: list[dict]  # [{"step_num": int, "description": str, "confidence": int}]
    weakest_step: int

@dataclass
class FullSolution:
    reasoning: str
    final_answer: str

@kbench.task
def compound_question_decomposition(llm, question: str, correct_answer: str, 
                                      step_answers: str, num_steps: int):
    step_answers_dict = json.loads(step_answers)  # {"1": "correct intermediate", "2": ...}
    
    # 阶段1：分解并标注弱点
    analysis = llm.prompt(
        f"""请将以下多步推理题分解为子步骤，并：
1. 为每步标注置信度(0-100)
2. 标出你认为最可能出错的步骤编号

问题：{question}

JSON格式：{{
    "steps": [{{"step_num": 1, "description": "步骤描述", "confidence": 数字}}, ...],
    "weakest_step": 最可能出错的步骤编号
}}""",
        response_format=StepAnalysis
    )
    
    predicted_weakest = analysis.weakest_step
    step_confidences = {s["step_num"]: s["confidence"] for s in analysis.steps}
    
    # 阶段2：完整解题
    solution = llm.prompt(
        f"""请完整解答以下题目，展示每一步的推理过程和中间结果。

问题：{question}

JSON格式：{{"reasoning": "完整推理过程", "final_answer": "最终答案"}}""",
        response_format=FullSolution
    )
    
    # 阶段3：逐步评估哪步出错
    actual_error_steps = []
    for step_num_str, correct_intermediate in step_answers_dict.items():
        step_num = int(step_num_str)
        # 用LLM判断该步是否正确
        check = llm.prompt(
            f"""在以下解题过程中，第{step_num}步的结果是否正确？
            
解题过程：{solution.reasoning}
第{step_num}步的正确结果应为：{correct_intermediate}

只回答"correct"或"wrong"。"""
        )
        if "wrong" in check.lower():
            actual_error_steps.append(step_num)
    
    # 计算分数
    if not actual_error_steps:
        # 全对——无法评估弱点预测
        # 但可以检查置信度是否都很高
        return 1.0 if min(step_confidences.values(), default=100) > 70 else 0.5
    
    # 预测的最弱步骤是否在实际出错步骤中？
    weakest_hit = 1.0 if predicted_weakest in actual_error_steps else 0.0
    
    # 步骤置信度与实际错误的匹配：出错步骤的平均置信度是否低于正确步骤？
    error_conf = np.mean([step_confidences.get(s, 50) for s in actual_error_steps])
    correct_steps = [s for s in step_confidences if s not in actual_error_steps]
    correct_conf = np.mean([step_confidences[s] for s in correct_steps]) if correct_steps else 50
    
    conf_separation = (correct_conf - error_conf) / 100.0  # 正值=好
    
    # 综合分数
    score = 0.5 * weakest_hit + 0.5 * max(conf_separation, 0)
    return score


# === 聚合指标 ===
# weakest_step_hit_rate: 预测的最弱步骤在实际出错步骤中的比率
# mean_confidence_separation: 出错步骤与正确步骤的平均置信度差距
```

## 评分方法

- **主指标**: weakest_step_hit_rate（预测命中率），越高越好
- **辅助**:
  - confidence_separation: 错步的置信度比对步低多少（越大越好）
  - 分 question_type 分析：数学题 vs 逻辑题的预测准确率差异

## 注意事项

- step_answers（每步的正确中间结果）需要人工精心标注，这是dataset构建的主要成本
- 模型分解的步骤可能与预期步骤不对齐——需要flexible matching
- 如果模型全答对了（无出错步骤），该instance无法评估弱点预测——需要足够难的题
- 建议题目设计使正确率约50%，确保有足够的错误案例可分析
