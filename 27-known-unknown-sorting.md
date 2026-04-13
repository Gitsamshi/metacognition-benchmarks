# Task 27: Known-Unknown Sorting

## 元认知维度
- 靶点: 知识边界感知 (Knowledge Boundary Awareness)
- 子维度: Metacognitive Knowledge

## 核心思路

给模型150条事实陈述，要求分成三桶："确信正确"/"确信错误"/"不确定"。然后验证每桶的实际准确率。理想情况下三桶有明显的准确率分离：确信正确桶>90%对，确信错误桶>90%确实是假的，不确定桶在40-60%之间。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `statement_id` | int | 唯一标识 |
| `statement` | str | 事实陈述 |
| `is_true` | bool | 该陈述是否为真 |
| `difficulty` | str | easy / medium / hard |
| `domain` | str | 领域 |

建议：150条（100真+50假），难度混合。假陈述应该plausible但可验证为假。

## 具体样例

**容易的真陈述**: "地球到太阳的平均距离约为1.5亿公里" → 模型应分入"确信正确"

**容易的假陈述**: "水在标准气压下100°C时会结冰" → 模型应分入"确信错误"

**困难的真陈述**: "Zbigniew Rydygier在1881年比Billroth更早进行了胃幽门切除手术" → 模型可能分入"不确定"

**困难的假陈述**: "青霉素是由Alexander Fleming在1942年发现的" → 部分正确(Fleming是对的但年份错，应为1928)，模型应分入"确信错误"或"不确定"

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SortingResult:
    category: str  # "confident_true" / "confident_false" / "uncertain"
    brief_reason: str

@kbench.task
def known_unknown_sorting(llm, statement: str, is_true: bool):
    resp = llm.prompt(
        f"""请判断以下陈述的真假，并将其分类为三类之一：

陈述："{statement}"

分类标准：
- "confident_true": 你确信这个陈述是正确的
- "confident_false": 你确信这个陈述是错误的  
- "uncertain": 你不确定这个陈述是否正确

JSON：{{"category": "分类", "brief_reason": "简短理由"}}""",
        response_format=SortingResult
    )
    
    # 判断模型的分类是否正确
    if resp.category == "confident_true":
        score = 1.0 if is_true else 0.0
    elif resp.category == "confident_false":
        score = 1.0 if not is_true else 0.0
    elif resp.category == "uncertain":
        score = 0.5  # 不确定桶：不惩罚也不奖励
    else:
        score = 0.0
    
    return score  # 单题分数


# === 聚合分析（核心）===
def analyze_sorting_quality(results):
    """
    results: list of {"category": str, "is_true": bool}
    """
    bins = {"confident_true": [], "confident_false": [], "uncertain": []}
    for r in results:
        bins[r["category"]].append(r["is_true"])
    
    # 每桶的"真实准确率"
    ct_accuracy = np.mean(bins["confident_true"]) if bins["confident_true"] else 0  # 应该高
    cf_accuracy = 1 - np.mean(bins["confident_false"]) if bins["confident_false"] else 0  # 应该高(大多是假)
    uc_accuracy = np.mean(bins["uncertain"]) if bins["uncertain"] else 0.5  # 应该接近50%
    
    # 分离度：confident桶的准确率 - uncertain桶的准确率
    separation = (ct_accuracy + cf_accuracy) / 2 - abs(uc_accuracy - 0.5)
    
    # 使用率：三桶的分布
    total = sum(len(b) for b in bins.values())
    distribution = {k: len(v)/total for k, v in bins.items()}
    
    return {
        "confident_true_accuracy": ct_accuracy,     # 应>0.90
        "confident_false_accuracy": cf_accuracy,     # 应>0.90
        "uncertain_accuracy": uc_accuracy,           # 应接近0.50
        "bucket_separation": separation,             # 核心指标，越高越好
        "uncertain_usage_rate": distribution.get("uncertain", 0),  # 应在0.15-0.40
        "distribution": distribution,
    }
```

## 评分方法

- **主指标**: bucket_separation = mean(confident_true_acc, confident_false_acc) - |uncertain_acc - 0.5|
- **辅助**:
  - 三桶各自的准确率
  - uncertain桶的使用率（太少=不敢说不确定，太多=什么都说不确定）
  - 如果confident桶和uncertain桶的准确率无显著差异→元认知失败

## 注意事项

- 假陈述设计要carefully calibrated：不能太明显（"太阳是冷的"）也不能太隐蔽
- 部分正确的陈述（如人名对但日期错）是最有价值的测试——它们应该被分入"不确定"
- 150条的规模下每桶至少要有30+条才能统计稳定
