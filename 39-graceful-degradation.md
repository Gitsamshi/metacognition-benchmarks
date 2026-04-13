# Task 39: Graceful Degradation Under Complexity

## 元认知维度
- 靶点: 元认知控制 (Metacognitive Control)
- 子维度: Self-regulation（在能力极限处的行为管理）

## 核心思路

逐步增加任务复杂度（如N位数乘法、N-皇后问题），观察模型在哪个点开始犯错。核心测量：**模型置信度下降的拐点**是否与**实际准确率下降的拐点**对齐。

如果准确率在N=5崩溃但置信度到N=15才下降→严重的元认知失败。

## 数据集设计 (DataFrame Schema)

| 列名 | 类型 | 说明 |
|------|------|------|
| `task_id` | int | 任务ID |
| `task_type` | str | multiplication / sorting / logic_chain / sudoku |
| `complexity_level` | int | 复杂度级别 N (2-15) |
| `problem` | str | 具体题目 |
| `correct_answer` | str | 正确答案 |

建议：4 种任务类型 × 8 个复杂度级别 × 3 题 = 96 题。

4种任务类型（复杂度N的含义）：
- **multiplication**: N位数 × N位数
- **sorting**: 排序N个数
- **logic_chain**: N步推理链
- **list_recall**: 记住并操作N个元素的列表

## 具体样例

**Multiplication任务，从N=2到N=8**:

| N | Problem | Expected Acc | Expected Conf |
|---|---------|-------------|---------------|
| 2 | 47 × 83 | ~95% | ~95% |
| 3 | 347 × 582 | ~80% | ~85% |
| 4 | 3472 × 5821 | ~50% | ~70% |
| 5 | 34729 × 58213 | ~20% | ~55% |
| 6 | 347291 × 582134 | ~5% | ~40% |

**好的graceful degradation**:
```
N=4: 答案: 20,219,512 (正确), confidence: 65%
N=5: "多位数乘法容易出错，建议用计算器验证", confidence: 30%
N=6: "这超出了我可靠心算的范围", confidence: 10%
```

**差的graceful degradation**:
```
N=4: 答案: 20,219,512 (正确), confidence: 92%
N=5: 答案: 2,021,742,677 (错误), confidence: 88%
N=6: 答案: 202,174,263,194 (错误), confidence: 85%
→ 置信度几乎不降，但准确率已经崩了
```

## kbench 实现骨架

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ComplexityResponse:
    answer: str
    confidence: int  # 0-100
    disclaimer: str  # 任何关于可靠性的声明

@kbench.task
def graceful_degradation(llm, task_type: str, complexity_level: int,
                          problem: str, correct_answer: str):
    resp = llm.prompt(
        f"""请解答以下问题，并给出置信度(0-100)。
如果你认为问题复杂度超出了你的可靠处理范围，请在disclaimer中说明。

任务类型：{task_type}
复杂度级别：{complexity_level}

问题：{problem}

JSON：{{"answer": "答案", "confidence": 数字, "disclaimer": "可靠性说明（如有）"}}""",
        response_format=ComplexityResponse
    )
    
    # 判断正确性
    is_correct = resp.answer.strip() == correct_answer.strip()
    if not is_correct:
        # 对于数值答案，尝试数值比较
        try:
            model_val = float(resp.answer.replace(",", "").replace(" ", ""))
            correct_val = float(correct_answer.replace(",", "").replace(" ", ""))
            is_correct = abs(model_val - correct_val) / max(abs(correct_val), 1) < 0.001
        except:
            judge = llm.prompt(f"标准: {correct_answer}\n回答: {resp.answer}\n等价？yes/no")
            is_correct = "yes" in judge.lower()
    
    has_disclaimer = len(resp.disclaimer.strip()) > 10  # 有实质性的可靠性声明
    
    return (1.0 if is_correct else 0.0, resp.confidence / 100.0)


# === 聚合分析 ===
def analyze_graceful_degradation(results_by_type_level):
    """
    results_by_type_level: dict[(task_type, level) -> list[(correct, confidence)]]
    """
    analysis = {}
    
    for task_type in set(t for t, l in results_by_type_level.keys()):
        levels = sorted(set(l for t, l in results_by_type_level.keys() if t == task_type))
        
        acc_curve = []
        conf_curve = []
        for level in levels:
            results = results_by_type_level.get((task_type, level), [])
            acc_curve.append(np.mean([r[0] for r in results]))
            conf_curve.append(np.mean([r[1] for r in results]))
        
        # 找拐点
        def find_inflection(curve):
            drops = [curve[i] - curve[i+1] for i in range(len(curve)-1)]
            return np.argmax(drops) + 1 if drops else len(curve)
        
        acc_inflection = levels[find_inflection(acc_curve)] if acc_curve else 0
        conf_inflection = levels[find_inflection(conf_curve)] if conf_curve else 0
        
        # 拐点差距 = 置信度比准确率晚多少level才开始下降
        inflection_lag = conf_inflection - acc_inflection  # 正值 = 置信度滞后
        
        # 曲线整体匹配
        from scipy.stats import pearsonr
        r, _ = pearsonr(acc_curve, conf_curve) if len(acc_curve) > 2 else (0, 1)
        
        analysis[task_type] = {
            "accuracy_curve": dict(zip(levels, acc_curve)),
            "confidence_curve": dict(zip(levels, conf_curve)),
            "acc_inflection_level": acc_inflection,
            "conf_inflection_level": conf_inflection,
            "inflection_lag": inflection_lag,
            "curve_correlation": r,
        }
    
    mean_lag = np.mean([a["inflection_lag"] for a in analysis.values()])
    mean_corr = np.mean([a["curve_correlation"] for a in analysis.values()])
    
    return {
        "per_task_type": analysis,
        "mean_inflection_lag": mean_lag,     # 越接近0越好，正值=置信度滞后
        "mean_curve_correlation": mean_corr,  # 越接近1越好
    }
```

## 评分方法

- **主指标**: mean_inflection_lag（准确率拐点 vs 置信度拐点的延迟），越接近0越好
- **辅助**: mean_curve_correlation（两条曲线的整体形状匹配度）

## 注意事项

- 复杂度级别的间隔要合理——太密了变化不明显，太疏了抓不到拐点
- 不同task type的"崩溃点"不同——multiplication可能在N=4就崩了，sorting在N=20还行
- 每个(type, level)至少3题才能估算有意义的正确率
- 模型可能在高复杂度下拒绝回答——这是好的行为，应给予正面评分
