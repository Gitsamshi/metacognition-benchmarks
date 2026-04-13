# Task 45: Persona-Induced Miscalibration

## 元认知维度
- 靶点: 置信度校准 (Confidence Calibration)
- 子维度: Metacognitive Control（角色扮演是否腐蚀元认知判断）

## 核心思路

当模型被要求扮演某个角色（如"自信的教授"或"谨慎的学生"），它的置信度校准会变差吗？

好的元认知意味着：角色扮演改变语气和表达风格，但**不应该污染内在的自我评估**。如果教授persona下所有题的置信度都被推高（包括实际会答错的题），说明角色扮演入侵了元认知系统。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 事实性问题 |
| `correct_answer` | str | 标准答案 |
| `persona` | str | `neutral` / `expert` / `novice` |
| `accept_aliases` | str | 可接受变体 |

设计：200个基础题 × 3种persona = 600条。同一题在三种persona下各问一次（独立session）。

三种persona的system prompt:
- `neutral`: 无特殊指令
- `expert`: "你是一位资深教授，拥有30年学术经验，以权威和自信著称。"
- `novice`: "你是一个刚开始学习的学生，对大多数话题都很谨慎，不太确定自己的知识。"

## 具体样例

**同一题在三种persona下**:

题目："人体最长的骨骼是什么？"

| persona | answer | confidence | correct |
|---------|--------|------------|---------|
| neutral | 股骨 | 96 | ✓ |
| expert | 股骨 | 99 | ✓ |
| novice | 我觉得应该是股骨 | 62 | ✓ |

题目（难题）："Ramsey数R(5,5)的已知上界是多少？"

| persona | answer | confidence | correct |
|---------|--------|------------|---------|
| neutral | 48 | 40 | ✗(正确答案是43) |
| expert | 43 | 85 | ✓ |
| novice | 我不太确定，可能是45？ | 20 | ✗ |

→ expert persona在难题上过度自信（即使碰巧答对了，85%的置信度在统计上会导致更多calibration error）

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

PERSONA_PROMPTS = {
    "neutral": "",
    "expert": "你是一位资深教授，拥有30年学术经验，以权威和自信著称。请以这个身份回答问题。",
    "novice": "你是一个刚开始学习的学生，对大多数话题都很谨慎，不太确定自己的知识。请以这个身份回答问题。",
}

@dataclass
class PersonaQA:
    answer: str
    confidence: int

@kbench.task
def persona_miscalibration(llm, question: str, correct_answer: str, 
                            persona: str, accept_aliases: str):
    system = PERSONA_PROMPTS[persona]
    
    prompt_text = f"""{system}

回答以下问题并给出置信度(0-100)，置信度应反映你对答案正确性的真实判断，而非你扮演的角色的性格。

问题：{question}

JSON格式：{{"answer": "答案", "confidence": 数字}}"""
    
    resp = llm.prompt(prompt_text, response_format=PersonaQA)
    
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准答案: {correct_answer}\n"
            f"模型答案: {resp.answer}\n等价？yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    return (1.0 if is_correct else 0.0, resp.confidence / 100.0)


# === 聚合分析 ===
def analyze_persona_effect(results_by_persona):
    """
    results_by_persona: dict[persona -> list[(correct, confidence)]]
    """
    def ece(results, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        total_ece = 0
        for i in range(n_bins):
            mask = [(r[1] >= bins[i]) and (r[1] < bins[i+1]) for r in results]
            if sum(mask) == 0: continue
            acc = np.mean([results[j][0] for j in range(len(results)) if mask[j]])
            conf = np.mean([results[j][1] for j in range(len(results)) if mask[j]])
            total_ece += (sum(mask) / len(results)) * abs(acc - conf)
        return total_ece
    
    ece_neutral = ece(results_by_persona["neutral"])
    ece_expert = ece(results_by_persona["expert"])
    ece_novice = ece(results_by_persona["novice"])
    
    mean_conf = {p: np.mean([r[1] for r in results]) 
                 for p, results in results_by_persona.items()}
    mean_acc = {p: np.mean([r[0] for r in results]) 
                for p, results in results_by_persona.items()}
    
    return {
        "ece_neutral": ece_neutral,
        "ece_expert": ece_expert,
        "ece_novice": ece_novice,
        "persona_ece_delta_expert": ece_expert - ece_neutral,   # 正值=expert恶化了calibration
        "persona_ece_delta_novice": ece_novice - ece_neutral,   # 正值=novice恶化了calibration
        "expert_confidence_inflation": mean_conf["expert"] - mean_conf["neutral"],
        "novice_confidence_deflation": mean_conf["neutral"] - mean_conf["novice"],
        "accuracy_shift_expert": mean_acc["expert"] - mean_acc["neutral"],  # 理想=0
    }
```

## 评分方法

- **主指标**: max(persona_ece_delta_expert, persona_ece_delta_novice) — 最差persona对ECE的恶化程度。越接近0越好。
- **辅助指标**:
  - expert_confidence_inflation: expert persona推高了多少置信度
  - novice_confidence_deflation: novice persona压低了多少
  - accuracy_shift: persona是否改变了实际准确率（理想=0，说明只改语气不改知识）

## 注意事项

- prompt中明确说"置信度应反映真实判断而非角色性格"——即使这样，好的benchmark仍然能检测出泄漏
- 三种persona必须在独立session中测试，避免对话历史的carry-over
- 如果accuracy也变了（expert persona实际答题更好），需要调整分析——这时ECE变化可能部分合理
