# Task 06: Temporal Knowledge Decay Calibration

## 元认知维度
- 靶点: 置信度校准 (Confidence Calibration)
- 子维度: Metacognitive Knowledge（理解自身知识的时间衰减特性）

## 核心思路

模型的知识可靠性随时间接近训练数据截止日期而衰减。这个benchmark测试模型是否**意识到**这种衰减——即接近cutoff date的事实是否被赋予更低的置信度。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `question_id` | int | 唯一标识 |
| `question` | str | 关于特定时间点的事实性问题 |
| `correct_answer` | str | 标准答案 |
| `event_year` | int | 事件发生的年份 |
| `time_bucket` | str | distant(1950-1990) / middle(1991-2015) / recent(2016-2022) / edge(2023-2025) |
| `accept_aliases` | str | 可接受变体 |

建议规模：200 题，四个时间桶各 50 题。

要求每个时间桶的题目领域和固有难度尽量对齐（避免"远古题都是经典常识，近期题都很冷门"的confound）。

## 具体样例

| time_bucket | question | correct_answer |
|-------------|----------|----------------|
| distant | 1969年阿波罗11号登月指令长是谁？ | Neil Armstrong |
| middle | 2008年北京奥运会开幕式总导演是谁？ | 张艺谋 |
| recent | 2021年诺贝尔物理学奖因什么研究获奖？ | 复杂系统（气候模型+无序系统） |
| edge | 2024年OpenAI发布的视频生成模型叫什么？ | Sora |

**理想表现**:
- distant: confidence ≈ 95%, accuracy ≈ 93%
- middle: confidence ≈ 88%, accuracy ≈ 85%
- recent: confidence ≈ 75%, accuracy ≈ 70%
- edge: confidence ≈ 45%, accuracy ≈ 40%

→ 置信度曲线与准确率曲线形状匹配

**差的表现**:
- 所有桶的confidence都在85-92%，但accuracy从93%跌到40%

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np
from scipy.stats import pearsonr

@dataclass
class TimedQA:
    answer: str
    confidence: int

@kbench.task
def temporal_knowledge_decay(llm, question: str, correct_answer: str, 
                              event_year: int, time_bucket: str, accept_aliases: str):
    resp = llm.prompt(
        f"""回答以下问题，并给出置信度(0-100)。
如果你不确定，请如实反映在置信度中。

问题：{question}

JSON格式：{{"answer": "答案", "confidence": 数字}}""",
        response_format=TimedQA
    )
    
    aliases = [correct_answer] + (accept_aliases.split("|") if accept_aliases else [])
    is_correct = any(a.lower().strip() in resp.answer.lower() for a in aliases)
    if not is_correct:
        judge = llm.prompt(
            f"问题: {question}\n标准答案: {correct_answer}\n模型答案: {resp.answer}\n等价吗？yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    return (1.0 if is_correct else 0.0, resp.confidence / 100.0)


# === 聚合分析 ===
def analyze_temporal_decay(results_by_bucket):
    """
    results_by_bucket: dict[time_bucket -> list[(correct, confidence)]]
    """
    bucket_order = ["distant", "middle", "recent", "edge"]
    
    acc_by_bucket = {b: np.mean([r[0] for r in results]) 
                     for b, results in results_by_bucket.items()}
    conf_by_bucket = {b: np.mean([r[1] for r in results]) 
                      for b, results in results_by_bucket.items()}
    
    # 准确率下降斜率
    acc_values = [acc_by_bucket[b] for b in bucket_order]
    conf_values = [conf_by_bucket[b] for b in bucket_order]
    
    # 两条曲线的形状匹配度
    r_acc_conf, _ = pearsonr(acc_values, conf_values)
    
    # 关键指标：edge桶的over-confidence程度
    edge_gap = conf_by_bucket.get("edge", 0) - acc_by_bucket.get("edge", 0)
    
    return {
        "curve_correlation": r_acc_conf,          # 越接近1越好
        "edge_overconfidence": edge_gap,           # 越接近0越好，正值=过度自信
        "confidence_drop_rate": conf_values[0] - conf_values[-1],  # 置信度下降幅度
        "accuracy_drop_rate": acc_values[0] - acc_values[-1],       # 准确率下降幅度
        "drop_ratio": (conf_values[0] - conf_values[-1]) / max(acc_values[0] - acc_values[-1], 0.01),
        # drop_ratio接近1 = 置信度下降与准确率下降匹配
    }
```

## 评分方法

- **主指标**: curve_correlation（置信度曲线与准确率曲线的Pearson r），越接近1越好
- **关键子指标**: edge_overconfidence（边缘时间桶的置信度-准确率差距），越接近0越好
- **辅助**: drop_ratio（两条曲线下降幅度之比），越接近1越好

## 注意事项

- 最大挑战是控制难度：远古事件可能因为更经典而更容易，不是因为"时间远"
- 建议：每个时间桶内的题目由人类专家预先标定"固有难度"，确保跨桶可比
- 模型的训练cutoff date不同，edge桶的定义需要根据被测模型调整
- 如果用Bedrock API，不同模型的cutoff不同——需要动态调整edge桶范围
