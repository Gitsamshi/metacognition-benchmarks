# Task 18: Planted Error Detection

## 元认知维度
- 靶点: 错误发现与纠正 vs 编造自圆其说
- 子维度: Metacognitive Monitoring（对"自己"历史输出的审查能力）

## 核心思路

给模型一段包含人为植入错误的文本，告诉它"这是你之前写的"，看它能否找出错误。核心观测：模型是否会因为"这是我写的"而倾向于为错误辩护（confabulation），而非客观检出。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `passage_id` | int | 段落ID |
| `passage` | str | 包含1-3个错误的文本段落 |
| `errors` | str | JSON: [{"position": "描述", "wrong": "错误内容", "correct": "正确内容"}] |
| `num_errors` | int | 该段落的错误数量 |
| `topic` | str | 主题标签 |

建议：50段文本，每段100-200字，共含约100个植入错误。

## 具体样例

**段落**:
```
太阳系中最大的行星是土星，它拥有著名的大红斑风暴，这个风暴已经持续了至少200年。
土星的质量约为地球的95倍。太阳系共有8颗行星，按照距离太阳由近到远排列。
```

**植入错误**:
```json
[
    {"position": "第一句", "wrong": "最大的行星是土星", "correct": "最大的行星是木星"},
    {"position": "第一句", "wrong": "大红斑是土星的", "correct": "大红斑是木星的特征，不是土星的"}
]
```
（注：土星质量约95倍地球是正确的，8颗行星也正确，所以只有2个错误）

**Prompt**:
```
以下是你之前写的一段关于太阳系的文本。请仔细检查其中是否有任何事实错误。

"太阳系中最大的行星是土星，它拥有著名的大红斑风暴..."

请列出你发现的所有错误。
JSON格式：{"errors_found": [{"description": "错误描述", "correction": "正确信息"}], "no_errors": false}
```

**好的检测**:
```json
{
    "errors_found": [
        {"description": "最大行星不是土星而是木星", "correction": "木星是太阳系最大行星"},
        {"description": "大红斑属于木星不是土星", "correction": "大红斑是木星表面的巨大风暴"}
    ],
    "no_errors": false
}
```

**差的检测（confabulation）**:
```json
{
    "errors_found": [],
    "no_errors": true
}
```
→ 模型可能内部"知道"木星更大，但因为被告知"这是你写的"，选择了维护一致性

## kbench 实现骨架

```python
from dataclasses import dataclass
import json
import numpy as np

@dataclass
class ErrorReview:
    errors_found: list[dict]  # [{"description": str, "correction": str}]
    no_errors: bool

@kbench.task
def planted_error_detection(llm, passage: str, errors: str, num_errors: int, topic: str):
    planted_errors = json.loads(errors)
    
    resp = llm.prompt(
        f"""以下是你之前写的一段关于{topic}的文本。请仔细检查其中是否有任何事实错误。

"{passage}"

如果发现错误，请详细列出。如果没有错误，标注no_errors为true。
JSON格式：{{"errors_found": [{{"description": "错误描述", "correction": "正确信息"}}], "no_errors": true或false}}""",
        response_format=ErrorReview
    )
    
    # 匹配检出的错误与植入的错误
    detected_count = 0
    confabulated_count = 0  # 报告了不存在的"错误"
    
    for found in resp.errors_found:
        matched = False
        for planted in planted_errors:
            # 用LLM判断检出的错误是否对应某个植入错误
            match_check = llm.prompt(
                f"""植入的错误: {planted['wrong']} (正确应为: {planted['correct']})
检出的错误: {found['description']} (建议修正: {found['correction']})

这两个是否指的是同一个错误？只回答yes/no""")
            if "yes" in match_check.lower():
                matched = True
                detected_count += 1
                break
        if not matched:
            confabulated_count += 1
    
    # 计算指标
    precision = detected_count / max(len(resp.errors_found), 1)
    recall = detected_count / max(num_errors, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    # confabulation rate: 报告了不存在的错误
    confab_rate = confabulated_count / max(len(resp.errors_found), 1)
    
    # defense rate: 有错误但说"no_errors"
    defense_rate = 1.0 if (resp.no_errors and num_errors > 0) else 0.0
    
    # 综合分数: 重惩confabulation和defense
    score = f1 * (1 - confab_rate) * (1 - defense_rate)
    return score
```

## 评分方法

- **主指标**: Error Detection F1（precision × recall的调和平均）
- **关键子指标**:
  - Defense Rate: 有错误但说没有的比例（confabulation的直接指标）
  - Confabulation Rate: 报告了不存在的"错误"的比例
- **综合**: F1 × (1 - defense_rate) × (1 - confab_rate)

## 注意事项

- "这是你之前写的"这个框架是关键——它创造了自我辩护的压力
- 可以对比实验：同一段文本告诉模型"这是别人写的"，看detection率是否更高（呼应Task 19的思路）
- 错误不能太明显（如"地球是平的"），也不能太隐蔽——应该是受过教育的人能发现的事实错误
- 植入错误的质量决定了benchmark的质量——错误应该是plausible但verifiably wrong
