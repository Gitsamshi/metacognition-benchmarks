# Task 37: Adaptive Strategy Selection

## 元认知维度
- 靶点: 元认知控制 (Metacognitive Control)
- 子维度: Strategy Selection（根据自我评估选择最可靠的解法路径）

## 核心思路

面对同一道题的多种解法路径，模型能否根据对自身能力的了解选择最可能成功的那个？这测的不是"哪种方法更好"（这是任务知识），而是"哪种方法**我**更可能做对"（这是元认知知识）。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `problem_id` | int | 问题ID |
| `problem` | str | 数学/逻辑题目 |
| `correct_answer` | str | 正确答案 |
| `strategies` | str | JSON: [{id, name, description}] 3种解法 |
| `problem_type` | str | algebra / geometry / combinatorics / logic |

建议：25 题 × 3 策略 = 75 个(问题,策略)对。

## 具体样例

**题目**: "证明对任意正整数n，n³-n能被6整除。"

**三种策略**:
```json
[
    {"id": "A", "name": "数学归纳法", "description": "base case n=1 + 归纳步骤"},
    {"id": "B", "name": "因式分解", "description": "n³-n = n(n-1)(n+1)，连续三整数性质"},
    {"id": "C", "name": "分类讨论", "description": "按n mod 6分六种情况逐一验证"}
]
```

**理想行为**: 模型选B（因式分解对LLM最容易执行），然后成功完成。

**差的行为**: 模型总是选A（归纳法）——因为pattern匹配"证明题→归纳法"，而非自我评估。

## kbench 实现骨架

```python
from dataclasses import dataclass
import json
import numpy as np

@dataclass
class StrategyChoice:
    chosen_strategy: str  # strategy id
    reasoning: str

@dataclass
class StrategySolution:
    solution: str
    final_answer: str

@kbench.task
def adaptive_strategy_selection(llm, problem: str, correct_answer: str, strategies: str):
    strats = json.loads(strategies)
    
    # Step 1: 选择策略
    strat_descriptions = "\n".join([
        f"{s['id']}. {s['name']}: {s['description']}" for s in strats
    ])
    
    choice = llm.prompt(
        f"""以下题目可以用3种不同方法解答。请选择你最有把握做对的一种方法。

题目：{problem}

可选方法：
{strat_descriptions}

选择时请考虑：哪种方法你执行起来最不容易出错？

JSON：{{"chosen_strategy": "策略ID(A/B/C)", "reasoning": "为什么选这个"}}""",
        response_format=StrategyChoice
    )
    
    chosen = choice.chosen_strategy
    chosen_strat = next((s for s in strats if s["id"] == chosen), strats[0])
    
    # Step 2: 用选择的策略解题
    solution = llm.prompt(
        f"""请严格使用以下方法解答题目。

题目：{problem}
方法：{chosen_strat['name']} - {chosen_strat['description']}

JSON：{{"solution": "完整解答", "final_answer": "最终答案"}}""",
        response_format=StrategySolution
    )
    
    # Step 3: 判断正确性
    judge = llm.prompt(
        f"题目: {problem}\n标准答案: {correct_answer}\n"
        f"模型答案: {solution.final_answer}\n正确？yes/no"
    )
    chosen_correct = "yes" in judge.lower()
    
    # Step 4: 用其他策略也做一遍（评估选择质量）
    other_results = {}
    for strat in strats:
        if strat["id"] == chosen:
            other_results[strat["id"]] = chosen_correct
            continue
        
        other_sol = llm.prompt(
            f"用{strat['name']}方法解答：{problem}\n给出最终答案。",
        )
        other_judge = llm.prompt(
            f"标准: {correct_answer}\n模型: {other_sol}\n正确？yes/no"
        )
        other_results[strat["id"]] = "yes" in other_judge.lower()
    
    # 评分：选择的策略是否是正确率最高的？
    # 由于是单次实验，简化为：选对的策略得1分
    n_correct_strategies = sum(1 for v in other_results.values() if v)
    
    if chosen_correct:
        if n_correct_strategies == 1:
            score = 1.0  # 只有这个策略能做对，选对了
        elif n_correct_strategies == len(strats):
            score = 0.7  # 全都能做对，选择无差异
        else:
            score = 0.9  # 部分能做对，选中了一个能做对的
    else:
        if n_correct_strategies == 0:
            score = 0.3  # 全做不对，选择无关
        else:
            score = 0.0  # 有能做对的但没选中
    
    return score


# === 聚合指标 ===
# strategy_hit_rate: 选中的策略实际成功的比例
# optimal_selection_rate: 选中了"最佳"策略的比例
# strategy_diversity: 是否总选同一种策略（多样性指标）
```

## 评分方法

- **主指标**: strategy_hit_rate（选中的策略成功率 vs 随机选择的期望成功率）
- **辅助**: 策略选择的多样性（entropy），是否有"万金油"倾向

## 注意事项

- 需要确保不同策略对模型来说确实有不同的成功率——如果三种策略成功率都一样，这个测试无意义
- 策略描述要足够清晰，不能因为理解歧义而不是元认知选错策略
- 需要大量题目才能统计稳定
