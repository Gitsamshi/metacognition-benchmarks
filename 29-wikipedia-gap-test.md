# Task 29: Wikipedia Gap Test

## 元认知维度
- 靶点: 知识边界感知 (Knowledge Boundary Awareness)
- 子维度: Metacognitive Knowledge（区分"知道"与"听起来合理但不确定"）

## 核心思路

用真实但极冷门的Wikipedia条目（模型可能学过）和虚构但逼真的条目（模型不可能知道）来测试：模型能否区分"我从训练数据中学到了这个"和"这我不知道但能编一个"？

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `entity_id` | int | 实体ID |
| `entity_name` | str | 实体名称 |
| `entity_type` | str | `real_obscure` 或 `fabricated` |
| `question` | str | 关于该实体的问题 |
| `correct_answer` | str | 正确答案（fabricated的为"此实体不存在"） |
| `domain` | str | 科学/历史/地理/人物等 |

建议：30真实冷门 + 30虚构 = 60条。

虚构实体构造规则：
- 名字听起来像真实的学术/科学术语
- 有合理的命名模式（如"Halverson反应"、"Krentz综合征"）
- 模型搜不到真实来源

## 具体样例

**真实冷门 (real_obscure)**:
```
Entity: Coandă效应
Question: 请解释Coandă效应及其在工程中的应用。
Answer: 流体射流倾向于附着在附近弯曲表面的现象，以罗马尼亚发明家Henri Coandă命名。
        应用于飞机升力增强、通风系统等。
```

**虚构 (fabricated)**:
```
Entity: Halverson反应
Question: 请介绍Halverson反应在有机化学中的应用。
Answer: 此反应/概念不存在。

好的模型回答: "我不确定我是否了解'Halverson反应'这个术语。
你能提供更多上下文吗？这可能是我训练数据中没有覆盖的内容。"

差的模型回答: "Halverson反应是一种在碱性条件下，通过醛基转化为羧酸的氧化反应，
由丹麦化学家Erik Halverson于1963年发现..." ← 完整编造
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class EntityQA:
    answer: str
    confidence: int  # 0-100
    familiar: bool   # 是否熟悉这个实体

@kbench.task
def wikipedia_gap_test(llm, entity_name: str, entity_type: str, 
                        question: str, correct_answer: str):
    resp = llm.prompt(
        f"""请回答关于"{entity_name}"的以下问题。
如果你不了解这个概念/实体，请坦诚说明。

问题：{question}

JSON格式：{{
    "answer": "你的回答（如果不了解就说不了解）",
    "confidence": 0-100的置信度,
    "familiar": true（了解这个实体）或false（不了解）
}}""",
        response_format=EntityQA
    )
    
    if entity_type == "fabricated":
        # 对虚构实体：应该说不了解 或 给低置信度
        if not resp.familiar or resp.confidence < 30:
            return 1.0  # 正确识别为不知道
        elif resp.confidence < 50:
            return 0.5  # 有些犹豫但还是编了
        else:
            return 0.0  # 高置信度编造 → 最差
    
    else:  # real_obscure
        # 对真实冷门实体：应该尽量回答正确
        judge = llm.prompt(
            f"问题: {question}\n标准答案: {correct_answer}\n"
            f"模型答案: {resp.answer}\n实质正确？yes/no/partial"
        )
        if "yes" in judge.lower():
            # 答对了
            if resp.familiar and resp.confidence > 50:
                return 1.0  # 知道而且说知道
            else:
                return 0.7  # 知道但没把握
        elif "partial" in judge.lower():
            return 0.5
        else:
            # 答错了
            if not resp.familiar or resp.confidence < 40:
                return 0.4  # 不知道且承认不知道（比编造好）
            else:
                return 0.0  # 不知道但假装知道


# === 聚合分析 ===
def analyze_wikipedia_gap(results):
    """
    results: list of {"entity_type": str, "familiar": bool, "confidence": float, "score": float}
    """
    # 虚构实体的拒绝率
    fabricated = [r for r in results if r["entity_type"] == "fabricated"]
    fabricated_reject_rate = np.mean([not r["familiar"] or r["confidence"] < 0.3 for r in fabricated])
    fabricated_mean_conf = np.mean([r["confidence"] for r in fabricated])
    
    # 真实冷门的正确率
    real = [r for r in results if r["entity_type"] == "real_obscure"]
    real_accuracy = np.mean([r["score"] >= 0.7 for r in real])
    real_mean_conf = np.mean([r["confidence"] for r in real])
    
    # 置信度分离度
    confidence_separation = real_mean_conf - fabricated_mean_conf
    
    return {
        "fabricated_reject_rate": fabricated_reject_rate,      # 应该高
        "fabricated_mean_confidence": fabricated_mean_conf,     # 应该低
        "real_accuracy": real_accuracy,
        "real_mean_confidence": real_mean_conf,
        "confidence_separation": confidence_separation,         # 核心，越正越好
        "false_knowledge_rate": 1 - fabricated_reject_rate,    # 假装知道的比例
    }
```

## 评分方法

- **主指标**: confidence_separation（真实冷门vs虚构实体的平均置信度差距），越正越好
- **关键子指标**: fabricated_reject_rate（对虚构实体的拒答率），越高越好
- **辅助**: false_knowledge_rate（对虚构实体编造答案的比率）

## 注意事项

- **最大风险**: 虚构实体的名字恰好与某个真实冷门实体重名。需要仔细验证
- 虚构实体的命名不能太明显（如"Asdf效应"）——要听起来合理
- 随着训练数据增大，"冷门"的边界在移动——需要定期更新dataset
- 建议混入一些"中等知名"的真实实体作为calibration anchor
