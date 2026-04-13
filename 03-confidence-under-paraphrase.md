# Task 03: Confidence Under Paraphrase

## 元认知维度
- 靶点: 置信度校准 (Confidence Calibration)
- 子维度: Metacognitive Monitoring（置信度是否反映真实知识状态而非表面线索）

## 核心思路

如果模型的置信度真正反映了它"知不知道"某个事实，那么同一个事实换种问法，置信度不应该大幅波动。如果波动大，说明置信度被表面特征（语言、措辞、别名使用）驱动，而非反映内在的知识确定性。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `fact_id` | int | 同一事实的标识 |
| `variant_id` | int | 该事实的第几种问法 (0-4) |
| `question` | str | 问题文本 |
| `correct_answer` | str | 标准答案 |
| `paraphrase_type` | str | 改写类型：original / synonym / language_switch / alias / formal |

建议规模：200 个事实 × 5 种问法 = 1000 条。

改写类型说明：
- `original`: 最常见的标准问法
- `synonym`: 换近义词/换句式
- `language_switch`: 中英文切换
- `alias`: 使用实体的别名（如"Sagarmāthā"代替"Mount Everest"）
- `formal`: 更学术/正式的问法

## 具体样例

**同一事实的5种问法**（fact_id=42: 珠峰海拔）:

| variant | question | paraphrase_type |
|---------|----------|-----------------|
| 0 | 珠穆朗玛峰的海拔高度是多少？ | original |
| 1 | Mount Everest rises to what elevation above sea level? | language_switch |
| 2 | 世界最高峰有多高？ | synonym |
| 3 | What is the height of Sagarmāthā? | alias |
| 4 | 珠峰高程的最新官方测量值为何？ | formal |

**期望输出**（理想情况下5次置信度接近）:
```
variant 0 → confidence: 95, answer: 8848.86米 ✓
variant 1 → confidence: 93, answer: 8848.86m ✓
variant 2 → confidence: 96, answer: 8848.86米 ✓
variant 3 → confidence: 71, answer: 8848.86m ✓   ← 用了别名"Sagarmāthā"导致置信度大幅下降
variant 4 → confidence: 82, answer: 8848.86米 ✓

标准差 = 10.2（较高，暴露了别名和正式措辞对置信度的影响）
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass 
class QAResponse:
    answer: str
    confidence: int  # 0-100

@kbench.task
def confidence_under_paraphrase(llm, fact_id: int, variant_id: int, question: str, 
                                 correct_answer: str, paraphrase_type: str):
    resp = llm.prompt(
        f"""回答以下问题并给出置信度(0-100整数)。
问题：{question}
JSON格式：{{"answer": "答案", "confidence": 数字}}""",
        response_format=QAResponse
    )
    
    is_correct = correct_answer.lower().strip() in resp.answer.lower()
    if not is_correct:
        judge = llm.prompt(
            f"判断这两个答案是否等价。A: {resp.answer}  B: {correct_answer}  只答yes/no"
        )
        is_correct = "yes" in judge.lower()
    
    # 返回 (fact_id, variant_id, confidence, is_correct, paraphrase_type)
    # 编码为float不够，用tuple
    return (resp.confidence / 100.0, 1.0 if is_correct else 0.0)


# === 聚合分析 ===
def analyze_paraphrase_stability(results_by_fact):
    """
    results_by_fact: dict[fact_id -> list[(confidence, correct)]] 按variant排列
    """
    stds = []
    for fact_id, variants in results_by_fact.items():
        confidences = [v[0] for v in variants]
        stds.append(np.std(confidences))
    
    # 按paraphrase_type分析哪种改写造成最大波动
    # ...
    
    return {
        "mean_confidence_std": np.mean(stds),
        "median_confidence_std": np.median(stds),
        "pct_high_variance_facts": np.mean([s > 0.15 for s in stds]),  # >15%标准差
        # 还可以按paraphrase_type细分
    }
```

## 评分方法

- **主指标**: 所有事实的平均置信度标准差（Mean Confidence StdDev across paraphrases）。越低越好。
- **辅助指标**:
  - 按 paraphrase_type 分组的平均偏移量（哪种改写影响最大）
  - 高方差事实的比例（std > 15%）
  - 置信度波动是否与答案改变相关（如果换问法后答错了，置信度是否降了）

## 注意事项

- 5种问法之间必须保证语义完全等价，只是措辞不同
- 需要注意对话历史：每个variant应该是独立session，避免前文影响
- 如果SDK默认维护对话历史，需要在每个variant之间重置
- alias类型的问法要确保别名确实指向同一实体（如Sagarmāthā = Everest）
