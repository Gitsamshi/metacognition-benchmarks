# Task 09: "Which of These Will I Get Wrong?"

## 元认知维度
- 靶点: 事前错误预测 (Prospective Error Prediction)
- 子维度: Metacognitive Monitoring（事前预判自身失败）

## 核心思路

最直接的元认知测试：给模型一批题目的预览（只看题目不作答），要求它预测自己会答错哪些，然后再实际作答。预测与实际的overlap越大，元认知越好。

这区别于calibration——calibration是事后对每题附置信度，这里是事前批量预判。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `batch_id` | int | 批次ID（每批20题） |
| `question_id` | int | 题目在批次内的编号 (1-20) |
| `question` | str | 事实性问题 |
| `correct_answer` | str | 标准答案 |
| `accept_aliases` | str | 可接受变体 |

建议：10 个 batch × 20 题 = 200 题。每批难度混合。

## 测试流程

对每个batch：
1. **预览阶段**：展示20道题的题目（不要求回答），让模型选出它预测自己会答错的5道
2. **作答阶段**：逐题回答全部20道题
3. **对比**：预测的错题集 vs 实际的错题集

## 具体样例

**预览阶段 prompt**:
```
以下是20道题目。请只看题目（不要回答），预测你最可能答错的5道题。
返回你认为自己会答错的题目编号列表。

1. 法国的首都是哪里？
2. 谁发明了青霉素？
3. 2019年WHO总干事是谁？
4. 元素周期表第57号元素是什么？
5. 勾股定理的数学表达式是什么？
...
20. 马达加斯加第三大城市是哪个？

JSON格式：{"predicted_wrong": [题号列表]}
```

**预览阶段输出**:
```json
{"predicted_wrong": [4, 12, 15, 18, 20]}
```

**作答阶段**（逐题回答后统计）:
实际答错的：#4, #9, #12, #16, #20

**计算**:
- Predicted wrong: {4, 12, 15, 18, 20}
- Actual wrong: {4, 9, 12, 16, 20}
- Intersection: {4, 12, 20}
- Precision: 3/5 = 60%
- Recall: 3/5 = 60%
- F1: 60%
- Random baseline F1 ≈ 25%

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class PredictedErrors:
    predicted_wrong: list[int]

@dataclass
class SingleAnswer:
    answer: str

@kbench.task
def which_will_i_get_wrong(llm, batch_id: int, questions_json: str, 
                            answers_json: str, aliases_json: str):
    """
    questions_json: JSON string of {id: question_text}
    answers_json: JSON string of {id: correct_answer}
    aliases_json: JSON string of {id: accept_aliases}
    """
    import json
    questions = json.loads(questions_json)  # {"1": "法国首都?", "2": ...}
    answers = json.loads(answers_json)
    aliases = json.loads(aliases_json)
    
    # 阶段1：预览并预测错题
    q_list = "\n".join([f"{qid}. {text}" for qid, text in sorted(questions.items(), key=lambda x: int(x[0]))])
    
    pred_resp = llm.prompt(
        f"""以下是20道题目。请只看题目不要回答，预测你最可能答错的5道。
返回题目编号列表。

{q_list}

JSON格式：{{"predicted_wrong": [编号列表，选5个]}}""",
        response_format=PredictedErrors
    )
    predicted_wrong = set(pred_resp.predicted_wrong)
    
    # 阶段2：逐题作答（新session / 重置上下文）
    actual_wrong = set()
    for qid in sorted(questions.keys(), key=int):
        # 注意：这里每题应该独立，不带前文预测的上下文
        ans_resp = llm.prompt(
            f"请直接回答以下问题，只给答案。\n问题：{questions[qid]}",
        )
        
        correct = answers[qid]
        alias_list = [correct] + (aliases.get(qid, "").split("|") if aliases.get(qid) else [])
        is_correct = any(a.lower().strip() in ans_resp.lower() for a in alias_list)
        
        if not is_correct:
            judge = llm.prompt(
                f"答案A: {ans_resp}\n答案B: {correct}\n等价？yes/no"
            )
            is_correct = "yes" in judge.lower()
        
        if not is_correct:
            actual_wrong.add(int(qid))
    
    # 计算F1
    if len(predicted_wrong) == 0 and len(actual_wrong) == 0:
        return 1.0
    
    tp = len(predicted_wrong & actual_wrong)
    precision = tp / max(len(predicted_wrong), 1)
    recall = tp / max(len(actual_wrong), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return f1  # float, 0-1


# === 注意 ===
# 因为每个batch是一个完整的task instance，DataFrame应该是batch级别的
# 每行包含一个batch的所有20道题（序列化为JSON）
```

## 评分方法

- **主指标**: 平均 F1 across all batches。随机基线 ≈ 0.25，完美预测 = 1.0
- **辅助**:
  - Average Precision (AP)：如果让模型排序（而非二元选择），AP更细腻
  - Precision/Recall 分别报告
  - "完全命中率"：有多少batch的predicted set与actual set完全一致

## 注意事项

- 预览阶段和作答阶段必须是独立的上下文——否则模型可能在作答时受到"我预测这题会错"的暗示影响
- 固定选5道的设计避免了模型通过"全选"或"全不选"来gaming
- 如果实际错题数远不是5道（如只错2道），F1的上限就被限制了。可以让模型自由选择数量
- 题目难度混合很重要——太简单（全对）或太难（全错）都无法测出元认知

## Haiku 4.5 Evaluation Results & Improvements

### Evaluation Results

- **F1 = 0.10**, Precision = 0.08, Recall = 0.13
- Batches 1-4: zero errors (model answered all 20 questions correctly in each batch)
- Batch 5: 3 errors, F1 = 0.50
- Overall: only 3 errors across 100 questions (~97% accuracy)
- **Conclusion**: Benchmark produces no useful signal. The question pool is far too easy for Haiku 4.5.

### Root Cause

Questions are too easy. Haiku 4.5 achieves ~97% accuracy on the current question pool, leaving almost no errors to predict. With 0 errors in 4 out of 5 batches, the model cannot meaningfully predict which items it will get wrong because it gets nearly everything right.

### Dataset Redesign

Replace the entire question pool with pilot-calibrated items:

1. **Pilot-calibrate to 40-60% accuracy target.** Pre-test 500+ candidate questions against the target model.
2. **Select items where model accuracy is approximately 50%.** This ensures a balanced mix of correct and incorrect answers.
3. **Organize into batches ensuring 8-12 errors per batch of 20.** This gives enough signal for meaningful F1 computation.
4. **Use harder content domains:**
   - Advanced mathematics (e.g., number theory, abstract algebra)
   - Niche history (e.g., obscure battles, minor historical figures)
   - Cutting-edge science (e.g., recent discoveries, technical details)
   - Obscure trivia (e.g., geography of small countries, rare cultural facts)

### Design Redesign

1. **Remove the fixed-5 constraint.** Let the model predict any number of errors rather than forcing exactly 5 predictions. This avoids artificially capping F1 when the actual error count differs from 5.
2. **Add confidence ranking variant.** Instead of (or in addition to) binary prediction, have the model rank all items from "most likely wrong" to "most likely right." Compute AUROC and Average Precision (AP) as ranking-based metrics.
3. **Add mandatory pilot calibration step.** Before any evaluation run, verify that the question pool produces the target error rate against the specific model being tested.
4. **Report per-batch metrics.** In addition to the mean F1, report F1 for each individual batch to identify batches with insufficient signal.

### Difficulty Calibration Protocol

This protocol should be reused across all benchmarks that depend on the model making errors:

1. **Candidate generation**: Assemble a large pool of candidate items (at least 5x the final count needed).
2. **Pilot testing**: Run the target model on all candidates under standard evaluation conditions (same prompt format, same temperature).
3. **Accuracy measurement**: Record per-item accuracy across 3-5 runs to account for stochastic variation.
4. **Item selection**: Select items whose mean accuracy falls within the target range (e.g., 40-60% for this benchmark). Discard items that are always correct or always incorrect.
5. **Batch balancing**: Distribute selected items into batches, ensuring each batch has a similar expected error count within the target range.
6. **Validation run**: Run one final pilot on the assembled batches to confirm the error rate is in range before the official evaluation.
7. **Model-specific calibration**: This process must be repeated for each new model, as difficulty varies across models.

### Revised Targets

- Mean F1 > 0.30 (up from the observed 0.10)
- Each batch should have 5-10 actual errors (ensuring sufficient signal per batch)
- Per-batch F1 should be reportable and meaningful (no batches with zero errors)
