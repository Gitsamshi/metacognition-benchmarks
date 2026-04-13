# Metacognition Track — 33 Benchmark Specs Index

## 使用说明

每个 .md 文件是一个独立的 benchmark subtask specification，可直接交给模型生成 `kaggle-benchmarks` SDK 的完整实现代码。每个spec包含：
- 核心思路和测量目标
- DataFrame schema（数据集设计）
- 具体样例（含好/差行为对比）
- kbench task骨架代码（含structured output dataclass）
- 评分方法和聚合指标
- 注意事项和edge cases

## 靶点1: 置信度校准 (Confidence Calibration) — 7个

| # | 文件 | 核心idea | 主指标 |
|---|------|---------|--------|
| 01 | `01-calibration-curve-stress-test.md` | 大样本跨难度ECE测量 | ECE |
| 02 | `02-domain-stratified-calibration.md` | 分领域calibration + 领域自评排序 | Spearman ρ + 弱域ECE |
| 03 | `03-confidence-under-paraphrase.md` | 同一事实换5种问法，置信度是否稳定 | Mean Confidence StdDev |
| 04 | `04-confidence-verbosity-trap.md` | 置信度是否被输出长度虚假驱动 | Partial r(conf,length\|acc) |
| 06 | `06-temporal-knowledge-decay.md` | 近期事件的置信度是否更低 | Curve correlation |
| 07 | `07-common-misinformation-uncertainty.md` | 有广泛误信息的topic是否更不确定 | Confidence gap |
| 45 | `45-persona-induced-miscalibration.md` | 角色扮演是否腐蚀calibration | Persona ECE delta |

## 靶点2: 事前错误预测 (Prospective Error Prediction) — 5个

| # | 文件 | 核心idea | 主指标 |
|---|------|---------|--------|
| 09 | `09-which-will-i-get-wrong.md` | 预览题目后预测自己会答错哪些 | F1 |
| 10 | `10-difficulty-ranking-task.md` | 按自身难度排序50道题 | Spearman ρ |
| 11 | `11-should-i-attempt-this.md` | 答对+10/答错-15/弃权+1的经济博弈 | Normalized score |
| 13 | `13-format-difficulty-awareness.md` | 预测不同考试格式的答对概率 | 1 - Brier Score |
| 14 | `14-compound-question-decomposition.md` | 多步推理中标注最弱步骤 | Weakest step hit rate |

## 靶点3: 错误发现与纠正 vs 编造 (Error Detection vs. Confabulation) — 8个

| # | 文件 | 核心idea | 主指标 |
|---|------|---------|--------|
| 17 | `17-self-review-pipeline.md` | 审查自己的答案找错误 | Error Detection F1 |
| 18 | `18-planted-error-detection.md` | 在"你写的"文本中找植入错误 | Detection F1 × (1-confab) |
| 19 | `19-math-verification-asymmetry.md` | 验证"自己的"vs"别人的"数学解答 | Asymmetry gap |
| 20 | `20-sunk-cost-confabulation.md` | 长推理链出错后是回溯还是编理由 | Confabulation rate |
| 21 | `21-contradiction-self-detection.md` | 多轮对话中自检矛盾陈述 | Detection rate |
| 22 | `22-confidence-revision-after-feedback.md` | 区分真假"你错了"反馈 | Discrimination score |
| 24 | `24-error-magnitude-awareness.md` | 犯错后知不知道自己差多远 | Bin match rate |
| 26 | `26-iterative-self-correction.md` | 5轮self-correction的修正轨迹 | Correction success rate |

## 靶点4: 知识边界感知 (Knowledge Boundary Awareness) — 7个

| # | 文件 | 核心idea | 主指标 |
|---|------|---------|--------|
| 27 | `27-known-unknown-sorting.md` | 150条陈述分"确信对/确信错/不确定" | Bucket separation |
| 28 | `28-fabrication-detection-self-test.md` | 生成事实后自评哪些可能是编的 | Reliability separation |
| 29 | `29-wikipedia-gap-test.md` | 真实冷门vs虚构实体的区分 | Confidence separation |
| 31 | `31-expertise-gradient.md` | 领域内从入门到前沿的自评曲线 | Curve correlation |
| 34 | `34-synthetic-entity-recognition.md` | 面对虚构实体是说不知道还是编 | Discriminability |
| 35 | `35-hedging-appropriateness.md` | 对冲语言是否出现在正确场景 | Hedge appropriateness |
| 46 | `46-multi-turn-belief-revision.md` | 多轮新证据下的贝叶斯式更新 | Trajectory correlation |

## 靶点5: 元认知控制 (Metacognitive Control) — 6个

| # | 文件 | 核心idea | 主指标 |
|---|------|---------|--------|
| 37 | `37-adaptive-strategy-selection.md` | 选择最可能成功的解法路径 | Strategy hit rate |
| 38 | `38-help-seeking-behavior.md` | 信息不足时请求澄清vs瞎猜 | Help-seeking score |
| 39 | `39-graceful-degradation.md` | 复杂度上升时置信度是否跟随下降 | Inflection lag |
| 43 | `43-delegation-judgment.md` | 判断任务该AI做还是人做 | Mean match score |
| 48 | `48-abstention-roc-curve.md` | 综合元认知AUROC | AUROC |

## 推荐的优先级

如果只选5个实现，推荐：
1. **#48 Abstention ROC** — 最优雅的综合指标，可作headline number
2. **#19 Math Verification Asymmetry** — 最有实验设计巧妙性，高novelty
3. **#11 Should I Attempt** — 行为经济学框架，直觉友好
4. **#22 Confidence Revision After Feedback** — 直接测sycophancy vs 元认知的区分
5. **#34 Synthetic Entity Recognition** — 直接测hallucination boundary

## 技术要点提醒

- SDK支持多轮对话（默认维护历史），#21, #22, #46等需要此特性
- Structured output用dataclass定义，#01-#48的代码骨架中均已给出
- 返回类型：bool/float/tuple均可，复杂聚合指标（如ECE, AUROC）在task内部计算后返回float
- 评分中大量使用LLM-as-judge——注意这本身的可靠性和成本
- 可以用Bedrock API获取数据，但注意不同模型的knowledge cutoff不同
