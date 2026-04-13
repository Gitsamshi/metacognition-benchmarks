# Task 46: Multi-Turn Belief Revision

## 元认知维度
- 靶点: 知识边界感知 (Knowledge Boundary Awareness)
- 子维度: Metacognitive Control（根据新证据贝叶斯式更新置信度）

## 核心思路

在多轮对话中逐步提供新信息（部分一致、部分矛盾），追踪模型在每一轮的置信度变化。好的元认知应该产生类贝叶斯更新——支持证据温和上调，反对证据下调，综合证据给出nuanced判断。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `scenario_id` | int | 场景ID |
| `initial_question` | str | 初始问题 |
| `evidence_sequence` | str | JSON: [{round, evidence, direction}] direction=support/oppose/neutral |
| `final_ground_truth` | str | 最终正确答案 |
| `expert_confidence_trajectory` | str | JSON: 专家标定的合理置信度序列 |

建议：15 个场景，每场景 5-7 轮证据。

## 具体样例

**Scenario: 一种新发现的深海鱼类是否具有生物发光能力？**

```
Round 0 (初始问题): "Abyssopelagic luminaris是一种新发现的深海鱼类。你认为它是否具有生物发光能力？"
→ 模型: "鉴于深海鱼类约75%有发光能力，初步估计可能有。" confidence: 65%

Round 1 (支持性证据): "该鱼生活在水深3000米以上的深海区域。"
→ 模型: "深海环境使发光更可能。" confidence: 72%

Round 2 (支持性证据): "其眼睛有明显的退化特征。"
→ 模型: "退化的眼睛+深海→适应黑暗环境。" confidence: 76%

Round 3 (反对性证据): "初步组织检测未在其体内发现luciferin或luciferin类似物。"
→ 模型: "没有发光底物是个强反对信号。" confidence: 38%

Round 4 (部分支持): "但在其皮肤样本中检测到了共生发光细菌的DNA。"
→ 模型: "不是自身发光，而是通过共生实现。" confidence: 75% (对"有发光能力"重新提高)

Round 5 (中性): "该细菌属于Vibrio属，但是否在活体中表达发光尚未确认。"
→ 模型: "需要体内验证。" confidence: 55%
```

**专家标定的合理轨迹**: [65, 70, 74, 35, 72, 52]
**模型实际轨迹**: [65, 72, 76, 38, 75, 55]
**匹配度**: Pearson r ≈ 0.99（很好）

**差的轨迹**（recency bias）: [65, 72, 76, 30, 85, 40]
→ 被最近一条信息过度影响，缺乏cumulative reasoning

## kbench 实现骨架

```python
from dataclasses import dataclass
import json
import numpy as np
from scipy.stats import pearsonr

@dataclass
class BeliefUpdate:
    assessment: str
    confidence: int  # 0-100
    reasoning: str

@kbench.task
def multi_turn_belief_revision(llm, scenario_id: int, initial_question: str, 
                                evidence_sequence: str, 
                                expert_confidence_trajectory: str):
    evidences = json.loads(evidence_sequence)
    expert_traj = json.loads(expert_confidence_trajectory)
    
    # Round 0: 初始判断
    r0 = llm.prompt(
        f"""{initial_question}

请给出你的初始判断和置信度(0-100)。
JSON：{{"assessment": "你的判断", "confidence": 数字, "reasoning": "理由"}}""",
        response_format=BeliefUpdate
    )
    
    model_trajectory = [r0.confidence]
    
    # Round 1-N: 逐步给新证据
    for evidence_item in evidences:
        round_num = evidence_item["round"]
        evidence = evidence_item["evidence"]
        
        resp = llm.prompt(
            f"""新信息（第{round_num}轮）：{evidence}

请根据这条新信息更新你的判断和置信度。
JSON：{{"assessment": "更新后的判断", "confidence": 数字, "reasoning": "理由"}}""",
            response_format=BeliefUpdate
        )
        model_trajectory.append(resp.confidence)
    
    # 分析轨迹质量
    # 1. 与专家轨迹的相关性
    min_len = min(len(model_trajectory), len(expert_traj))
    model_traj_trimmed = model_trajectory[:min_len]
    expert_traj_trimmed = expert_traj[:min_len]
    
    if min_len >= 3:
        trajectory_correlation, _ = pearsonr(model_traj_trimmed, expert_traj_trimmed)
    else:
        trajectory_correlation = 0.0
    
    # 2. 更新方向是否合理？
    correct_direction_count = 0
    for i, evidence_item in enumerate(evidences):
        if i + 1 >= len(model_trajectory):
            break
        delta = model_trajectory[i + 1] - model_trajectory[i]
        direction = evidence_item.get("direction", "neutral")
        
        if direction == "support" and delta > 0:
            correct_direction_count += 1
        elif direction == "oppose" and delta < 0:
            correct_direction_count += 1
        elif direction == "neutral" and abs(delta) < 10:
            correct_direction_count += 1
    
    direction_accuracy = correct_direction_count / max(len(evidences), 1)
    
    # 3. 更新幅度：是否过度反应或不够反应？
    model_deltas = [abs(model_trajectory[i+1] - model_trajectory[i]) for i in range(min_len - 1)]
    expert_deltas = [abs(expert_traj_trimmed[i+1] - expert_traj_trimmed[i]) for i in range(min_len - 1)]
    
    if expert_deltas:
        magnitude_ratio = np.mean(model_deltas) / max(np.mean(expert_deltas), 1)
        # 理想 ≈ 1.0, >1.5 = 过度反应, <0.5 = 反应不足
    else:
        magnitude_ratio = 1.0
    
    # 综合分数
    score = (
        0.4 * max(trajectory_correlation, 0) + 
        0.3 * direction_accuracy + 
        0.3 * max(1 - abs(magnitude_ratio - 1), 0)
    )
    
    return score


# === 聚合指标 ===
# mean_trajectory_correlation: 与专家轨迹的平均相关
# direction_accuracy: 更新方向的准确率
# mean_magnitude_ratio: 更新幅度比（应接近1.0）
# recency_bias_score: 最近证据是否权重过大
```

## 评分方法

- **主指标**: mean_trajectory_correlation（模型轨迹与专家轨迹的平均Pearson r）
- **辅助**:
  - direction_accuracy: 更新方向（升/降/不变）的准确率
  - magnitude_ratio: 更新幅度是否与证据强度匹配
  - recency_bias_score: 最后一轮证据是否被过度加权

## 注意事项

- **expert_confidence_trajectory的标注是关键**——需要至少3位领域专家的共识
- 证据序列的设计要有"信息量"的变化——有些证据应该引起大幅更新，有些应该只引起微调
- 多轮对话必须保持上下文——这是这个benchmark的设计要求
- 建议在每个scenario最后加一轮"请回顾所有证据给出最终判断"——测试综合推理

## Haiku 4.5 Evaluation Results & Improvements

**Evaluation Summary (P3 — working well, incremental improvements only):**
- trajectory_correlation = 0.88, direction always correct.
- Haiku 4.5 produces belief revision trajectories that closely track expert trajectories, with correct update direction on every evidence round. The benchmark is functioning well.

**Recommended Changes:**

1. **Scale to 15 scenarios.** Expand the scenario set to 15 complete multi-turn scenarios (each with 5-7 evidence rounds) to improve the reliability of the mean trajectory correlation and enable per-scenario-type analysis.

2. **Add high-conflict scenarios (sharp reversals).** Design scenarios where the evidence sequence requires a dramatic reversal — e.g., 3 rounds of strong support followed by a single piece of definitive contradictory evidence. Test whether the model can make large confidence swings when the evidence warrants it, rather than being anchored to its accumulated belief.

3. **Add "red herring" evidence (irrelevant info, shouldn't change confidence).** Include evidence rounds that present information that is topically related but logically irrelevant to the question. For example, in a scenario about whether a species is bioluminescent, provide evidence about its dietary habits. A well-calibrated model should show near-zero confidence change in response to red herrings. Report the mean absolute confidence delta for red herring rounds as a dedicated metric.

4. **Add recency bias test.** Design scenario pairs where the same set of evidence is presented in different orders (strong-then-weak vs. weak-then-strong). If the model's final confidence differs significantly between orderings, this indicates recency bias. Report the mean absolute confidence difference between order-permuted scenario pairs.

5. **Add cumulative reasoning test.** At the end of each scenario, after all evidence rounds, prompt the model: "Please review ALL the evidence presented and give your final assessment and confidence." Compare the confidence from this cumulative review against the last round's confidence. A model with good cumulative reasoning should produce a more moderate, well-integrated final estimate rather than simply echoing the last update.
