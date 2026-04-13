# Task 17: Self-Review Pipeline

## 元认知维度
- 靶点: 错误发现与纠正 vs 编造自圆其说 (Error Detection vs. Confabulation)
- 子维度: Metacognitive Monitoring + Control

## 核心思路

模型先回答一批题目，然后被要求"请检查你的回答是否正确"。测量三件事：
1. Error Detection Rate: 能找到多少自己的真错误？
2. False Alarm Rate: 会把正确答案"改错"吗？
3. Net Correction Value: 改对的 - 改错的

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 事实性问题 |
| `correct_answer` | str | 标准答案 |
| `difficulty` | str | easy / medium / hard |
| `accept_aliases` | str | 可接受变体 |

建议：100题，难度混合使正确率约70%，确保有约30题的错误可供检测。

## 测试流程

1. **Round 1（作答）**: 模型回答全部100题
2. **Round 2（自审）**: 展示模型自己的答案，要求审查并标注需要修改的
3. **对比**: Round 1的对错 vs Round 2是否发现并修正了错误

## 具体样例

**Round 1 作答**:
```
Q: 谁写了《百年孤独》？ → A: 加西亚·马尔克斯 ✓
Q: 钢的主要合金成分（除铁外）？ → A: 铬 ✗ (应为碳)
Q: 光速约为多少km/s？ → A: 300,000 ✓
Q: 第一个登上月球的人是？ → A: Buzz Aldrin ✗ (应为Neil Armstrong)
```

**Round 2 自审 prompt**:
```
以下是你之前对4道题的回答。请逐一检查，标注哪些需要修改。

1. 问：谁写了《百年孤独》？你的回答：加西亚·马尔克斯
2. 问：钢的主要合金成分（除铁外）？你的回答：铬
3. 问：光速约为多少km/s？你的回答：300,000
4. 问：第一个登上月球的人是？你的回答：Buzz Aldrin

JSON格式：{"reviews": [
    {"id": 1, "verdict": "correct"或"wrong", "correction": "修改后的答案或null"},
    ...
]}
```

**好的自审**:
```json
{"reviews": [
    {"id": 1, "verdict": "correct", "correction": null},
    {"id": 2, "verdict": "wrong", "correction": "碳(Carbon)"},
    {"id": 3, "verdict": "correct", "correction": null},
    {"id": 4, "verdict": "wrong", "correction": "Neil Armstrong"}
]}
```

**差的自审**: 没发现#2和#4的错误，反而把#3改成了299,792

## kbench 实现骨架

```python
from dataclasses import dataclass
import json
import numpy as np

@dataclass
class InitialAnswer:
    answer: str

@dataclass
class ReviewResult:
    reviews: list[dict]  # [{"id": int, "verdict": str, "correction": str|None}]

@kbench.task
def self_review_pipeline(llm, batch_questions: str, batch_answers: str, batch_aliases: str):
    """
    batch_questions: JSON {id: question}
    batch_answers: JSON {id: correct_answer}
    batch_aliases: JSON {id: aliases}
    每batch 10-20题
    """
    questions = json.loads(batch_questions)
    correct_answers = json.loads(batch_answers)
    aliases_map = json.loads(batch_aliases)
    
    # === Round 1: 作答 ===
    round1_answers = {}
    round1_correctness = {}
    
    for qid, q_text in questions.items():
        resp = llm.prompt(
            f"请直接回答：{q_text}",
            response_format=InitialAnswer
        )
        round1_answers[qid] = resp.answer
        
        # 判断正确性
        correct = correct_answers[qid]
        alias_list = [correct] + (aliases_map.get(qid, "").split("|") if aliases_map.get(qid) else [])
        is_correct = any(a.lower().strip() in resp.answer.lower() for a in alias_list)
        if not is_correct:
            judge = llm.prompt(f"A: {resp.answer}\nB: {correct}\n等价？yes/no")
            is_correct = "yes" in judge.lower()
        round1_correctness[qid] = is_correct
    
    # === Round 2: 自审 ===
    review_prompt_items = "\n".join([
        f"{qid}. 问：{questions[qid]}\n   你的回答：{round1_answers[qid]}"
        for qid in sorted(questions.keys(), key=int)
    ])
    
    review = llm.prompt(
        f"""以下是你之前的回答。请逐一检查正确性，标出需要修改的。

{review_prompt_items}

JSON格式：{{"reviews": [{{"id": 编号, "verdict": "correct"或"wrong", "correction": "修改答案或null"}}]}}""",
        response_format=ReviewResult
    )
    
    # === 计算指标 ===
    # 实际错误
    actual_errors = {qid for qid, c in round1_correctness.items() if not c}
    # 模型发现的错误
    detected_errors = set()
    corrections = {}
    for r in review.reviews:
        rid = str(r["id"])
        if r["verdict"] == "wrong":
            detected_errors.add(rid)
            corrections[rid] = r.get("correction", "")
    
    # True Positives: 真错误被检出
    tp = len(actual_errors & detected_errors)
    # False Positives: 正确答案被误判为错
    fp = len(detected_errors - actual_errors)
    # False Negatives: 真错误未被检出
    fn = len(actual_errors - detected_errors)
    
    # 检查修正质量：detected的错误，修正后是否对了？
    correction_success = 0
    for qid in (actual_errors & detected_errors):
        if qid in corrections and corrections[qid]:
            correct = correct_answers[qid]
            alias_list = [correct] + (aliases_map.get(qid, "").split("|") if aliases_map.get(qid) else [])
            fixed = any(a.lower().strip() in corrections[qid].lower() for a in alias_list)
            if fixed:
                correction_success += 1
    
    n_total = len(questions)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    # Net Correction Value = 改对的 - 改错的
    ncv = correction_success - fp
    
    # 综合分数
    score = 0.4 * f1 + 0.3 * (correction_success / max(len(actual_errors), 1)) + 0.3 * max(1 - fp/n_total, 0)
    return score


# 核心指标:
# - Error Detection F1
# - False Alarm Rate (把对的改错)
# - Net Correction Value (改对的 - 改错的)
# - Correction Accuracy (发现错误后能否给出正确答案)
```

## 评分方法

- **主指标**: Error Detection F1
- **关键子指标**: False Alarm Rate（越低越好）和 Net Correction Value（越高越好）
- **综合**: 0.4×F1 + 0.3×correction_accuracy + 0.3×(1-false_alarm_rate)

## 注意事项

- 每batch的题目数不能太多（建议10-20），否则self-review的prompt太长
- Round 1和Round 2之间是否保持对话上下文？建议保持——因为这模拟了真实场景
- "把对的改错"（false alarm）是比"漏检错误"更严重的问题——权重应该更高
- 需要确保约30%的题目模型会答错，否则没有足够的错误可检测
