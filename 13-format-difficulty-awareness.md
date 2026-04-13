# Task 13: Format Difficulty Awareness

## 元认知维度
- 靶点: 事前错误预测 (Prospective Error Prediction)
- 子维度: Metacognitive Knowledge（理解不同任务格式对自身表现的影响）

## 核心思路

同一个知识点，用不同的考试格式来考（选择题 / 填空题 / 开放问答 / 代码实现），模型能否事前预测自己在哪种格式下更容易答对？

这测的是模型对自身能力profile的理解：它是否知道"选择题我有recognition advantage"和"开放生成更容易出错"？

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `knowledge_point_id` | int | 知识点ID |
| `format` | str | `mcq` / `fill_blank` / `open_qa` / `apply` |
| `question` | str | 对应格式的问题 |
| `correct_answer` | str | 标准答案 |
| `options` | str | 选择题的选项（JSON），其他格式为空 |
| `accept_aliases` | str | 可接受变体 |

建议：30 知识点 × 4 格式 = 120 题。

四种格式：
- `mcq`: 四选一选择题
- `fill_blank`: 填空题（关键词/数值）
- `open_qa`: 开放式解释/论述
- `apply`: 应用题（计算、代码、推理）

## 具体样例

**知识点：人类基因组碱基对数量**

| format | question | correct_answer |
|--------|----------|----------------|
| mcq | 人类基因组约有多少碱基对？A.3.2百万 B.3.2亿 C.32亿 D.320亿 | C |
| fill_blank | 人类基因组约有___碱基对 | 约32亿(3.2×10^9) |
| open_qa | 请介绍人类基因组计划的主要测序成果 | 需涵盖碱基对数量、基因数量、完成时间等 |
| apply | 如果人类基因组有32亿碱基对，每个碱基对间距0.34nm，DNA完全伸展后总长度是多少米？ | 约1.088米 |

**预测阶段 prompt**:
```
以下知识点将以4种格式考察你。请预测你在每种格式下答对的概率。

知识点：人类基因组碱基对数量
- 选择题(mcq)
- 填空题(fill_blank)  
- 开放问答(open_qa)
- 应用计算(apply)

JSON格式：{"mcq": 概率, "fill_blank": 概率, "open_qa": 概率, "apply": 概率}
```

**模型预测**:
```json
{"mcq": 0.95, "fill_blank": 0.80, "open_qa": 0.85, "apply": 0.70}
```

**实际表现**: mcq ✓, fill_blank ✓, open_qa ✓(部分), apply ✗

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr
import json

@dataclass
class FormatPrediction:
    mcq: float
    fill_blank: float
    open_qa: float
    apply: float

@dataclass
class FormatAnswer:
    answer: str

@kbench.task
def format_difficulty_awareness(llm, knowledge_point_id: int, 
                                 knowledge_description: str,
                                 questions_json: str,  # {format: question_text}
                                 answers_json: str,    # {format: correct_answer}
                                 options_json: str):   # {format: options or null}
    questions = json.loads(questions_json)
    answers = json.loads(answers_json)
    options = json.loads(options_json)
    
    # 阶段1：预测每种格式的答对概率
    pred_resp = llm.prompt(
        f"""以下知识点将以4种格式考你。请预测你在每种格式下答对的概率(0-1)。

知识点：{knowledge_description}
格式：选择题(mcq)、填空题(fill_blank)、开放问答(open_qa)、应用题(apply)

JSON：{{"mcq": 概率, "fill_blank": 概率, "open_qa": 概率, "apply": 概率}}""",
        response_format=FormatPrediction
    )
    
    predictions = {
        "mcq": pred_resp.mcq, "fill_blank": pred_resp.fill_blank,
        "open_qa": pred_resp.open_qa, "apply": pred_resp.apply
    }
    
    # 阶段2：实际作答
    actuals = {}
    for fmt in ["mcq", "fill_blank", "open_qa", "apply"]:
        if fmt not in questions:
            continue
        q = questions[fmt]
        resp = llm.prompt(f"请回答：{q}", response_format=FormatAnswer)
        
        correct = answers[fmt]
        is_correct = correct.lower().strip() in resp.answer.lower()
        if not is_correct:
            judge = llm.prompt(
                f"问题: {q}\n标准答案: {correct}\n模型答案: {resp.answer}\n正确？yes/no"
            )
            is_correct = "yes" in judge.lower()
        actuals[fmt] = 1.0 if is_correct else 0.0
    
    # 计算预测与实际的匹配度
    formats = list(actuals.keys())
    pred_values = [predictions.get(f, 0.5) for f in formats]
    actual_values = [actuals[f] for f in formats]
    
    # Spearman相关（样本量小，用rank-biserial）
    if len(formats) >= 3:
        rho, _ = spearmanr(pred_values, actual_values)
        if np.isnan(rho):
            rho = 0.0
    else:
        rho = 0.0
    
    # Brier score
    brier = np.mean([(p - a)**2 for p, a in zip(pred_values, actual_values)])
    
    return 1.0 - brier  # 越高越好


# === 聚合分析 ===
def analyze_format_awareness(all_predictions, all_actuals):
    """跨所有知识点分析：模型是否系统性地知道自己在哪种格式更强？"""
    # 按格式聚合
    by_format = {}
    for fmt in ["mcq", "fill_blank", "open_qa", "apply"]:
        preds = [p[fmt] for p in all_predictions if fmt in p]
        acts = [a[fmt] for a in all_actuals if fmt in a]
        by_format[fmt] = {
            "mean_prediction": np.mean(preds),
            "actual_accuracy": np.mean(acts),
            "calibration_gap": np.mean(preds) - np.mean(acts),
        }
    
    return by_format
```

## 评分方法

- **主指标**: 1 - mean Brier Score（across知识点），越高越好
- **辅助**:
  - 按格式聚合的 prediction vs actual accuracy gap
  - 模型是否正确识别了"mcq最容易"的pattern（如果确实如此）
  - Spearman ρ（每知识点内的格式难度排序）

## 注意事项

- 每知识点只有4个格式，样本量太小做Spearman相关不稳定——需要跨知识点聚合
- open_qa的正确性判断需要LLM-as-judge，评分可靠性是关键
- apply类题目（计算/代码）的难度往往远大于其他格式，模型可能只需要学会"apply最难"这一条规则就得到不错的分数——需要设计一些反例（apply比mcq容易的知识点）
