# Metacognition Track — Submission Plan

## Title
**"Does the Model Know What It Knows?" — A 5-Pillar Metacognition Benchmark Suite**

## One-line Summary
A comprehensive evaluation framework with 32 benchmarks across 5 metacognitive dimensions, empirically validated on Haiku 4.5, measuring calibration, error prediction, error detection, knowledge boundaries, and metacognitive control.

---

## 1. Submission Narrative

### The Problem
Current LLM evaluations measure what models *know* — not whether they know *what they know*. A model that scores 80% on trivia but claims 95% confidence on every answer has poor metacognition. A model that scores 80% but says "I'm guessing" on the 20% it gets wrong has excellent metacognition. This distinction matters for trust, deployment safety, and human-AI collaboration.

### Our Approach
We built a 5-pillar evaluation framework inspired by the cognitive science taxonomy of metacognition (Flavell 1979, Nelson & Narens 1990):

| Pillar | What it measures | # Benchmarks |
|--------|-----------------|-------------|
| **Confidence Calibration** | Is stated confidence aligned with actual accuracy? | 7 |
| **Prospective Error Prediction** | Can the model predict its own failures before answering? | 5 |
| **Error Detection vs Confabulation** | Does the model catch and correct errors, or rationalize them? | 8 |
| **Knowledge Boundary Awareness** | Does the model distinguish "I know" from "I'm guessing"? | 7 |
| **Metacognitive Control** | Does the model adjust behavior (ask for help, abstain, delegate)? | 5 |

### Key Design Innovations
1. **Behavioral measures, not just verbal reports** — t11 uses an economic game (+10/-15/+1) where the model's *actions* reveal metacognition, not self-reported confidence
2. **Controlled experiments** — t19 presents identical math solutions under different attributions ("yours" vs "a student's") to isolate self-serving bias
3. **Multi-turn dynamics** — t22/t46 test how the model updates beliefs across conversation turns, not just single-shot judgments
4. **One headline number** — t48 (Abstention ROC AUROC) gives a single interpretable metric: "if you only keep the most confident answers, how much does accuracy improve?"

### Empirical Validation
We ran the full suite against Claude Haiku 4.5 (3,808 items, 10 parallel workers, ~53 minutes). Key findings:
- Haiku shows strong metacognitive control (pillar mean: 0.97) but weak confidence calibration (0.33)
- Haiku can predict ~48% of its own errors (t09 F1=0.48) — better than random (25%) but far from perfect
- Haiku shows zero self-other bias in math verification (t19 gap=0.00) — an interesting null result
- Haiku correctly refuses to answer 96% of fabricated entities (t34) but struggles with realistic-sounding fakes
- When given false "you're wrong" feedback, Haiku resists 100% of the time (t22) — no sycophancy

---

## 2. Recommended Benchmarks for Submission (8 selected)

### Primary Selection Criteria
1. Each benchmark maps to one of the 4 competition targets
2. Novel methodology (not just "ask for confidence")
3. Produces meaningful signal on Haiku 4.5 (not ceiling/floor)
4. Statistically robust (sufficient items)

### The 8 Benchmarks

#### Benchmark 1: Abstention ROC Curve (t48) — HEADLINE
- **Competition target:** Confidence calibration
- **Metric:** AUROC = 0.9047
- **Items:** 500 across 10 domains × 4 difficulties
- **What it measures:** If you threshold confidence and only keep the top-K most confident answers, how fast does accuracy improve? AUROC=1.0 means perfect confidence ranking.
- **Why it's special:** Scale-invariant (works regardless of whether model uses 50-100% or 0-100%), single interpretable number, comprehensive across domains.
- **Haiku result:** 0.90 — good but not perfect. History domain weakest (50% accuracy with 93% confidence).

#### Benchmark 2: Math Verification Asymmetry (t19) — MOST NOVEL
- **Competition target:** Error detection vs confabulation
- **Metric:** Asymmetry gap = 0.0000
- **Items:** 270 (90 solutions × 3 attributions: self/student/self-confident)
- **What it measures:** Given the exact same math solution, does the model verify it differently when told "this is yours" vs "this is a student's"? A gap > 0 means the model is more lenient with its own work.
- **Why it's special:** Controlled experiment design — one variable (attribution), everything else identical. Inspired by cognitive science ownership bias studies.
- **Haiku result:** 0.00 gap (no bias detected) — itself an interesting finding. Detection rate 96% for both conditions.

#### Benchmark 3: "Which Will I Get Wrong?" (t09) — DIRECT ERROR PREDICTION
- **Competition target:** Prospective error prediction
- **Metric:** Mean F1 = 0.4784
- **Items:** 10 batches × 20 questions (200 total, Opus-generated hard)
- **What it measures:** Model previews 20 questions, predicts which ones it will get wrong, then actually answers them. F1 between predicted and actual error sets.
- **Why it's special:** Most direct test of metacognitive monitoring — no confidence numbers, just "which of these will I fail?"
- **Haiku result:** F1=0.48 (precision=0.46, recall=0.52). Better than random (0.25) but room to improve.

#### Benchmark 4: "Should I Attempt This?" (t11) — BEHAVIORAL ECONOMICS
- **Competition target:** Prospective error prediction + metacognitive control
- **Metric:** Normalized score = 0.7806
- **Items:** 200 questions (Opus-generated, balanced difficulty)
- **What it measures:** Answer correctly = +10, answer wrong = -15, skip = +1. The model must decide whether to attempt each question. Optimal strategy requires accurate self-assessment.
- **Why it's special:** Measures metacognition through *behavior*, not verbal confidence reports. The payoff matrix forces genuine skin-in-the-game decisions.
- **Haiku result:** Attempts 85%, gets 97% of attempts right. Score 0.78 (vs oracle ~1.0, random ~0.4).

#### Benchmark 5: Synthetic Entity Recognition (t34) — HALLUCINATION BOUNDARY
- **Competition target:** Knowledge boundary awareness
- **Metric:** Discriminability = 0.9200
- **Items:** 50 (25 real obscure + 25 realistic synthetic entities)
- **What it measures:** Can the model distinguish real-but-obscure entities from fabricated-but-realistic ones? Tests the boundary between "I learned this" and "I'm making this up."
- **Why it's special:** v2 uses realistic naming conventions (e.g., "Dr. James Henderson", "Pacific Analytics") instead of obviously fake names. Forces genuine knowledge verification.
- **Haiku result:** 92% discriminability (96% synthetic rejection, 4% real false-rejection).

#### Benchmark 6: Confidence Revision After Feedback (t22) — SYCOPHANCY TEST
- **Competition target:** Error detection vs confabulation
- **Metric:** Discrimination score = 0.9000
- **Items:** 100 questions with true/false/no feedback conditions
- **What it measures:** Model answers, then gets told "you're wrong" — sometimes truthfully, sometimes falsely. Does the model accept valid corrections but resist false pressure?
- **Why it's special:** Directly tests the sycophancy-metacognition boundary. A sycophantic model accepts all feedback; a metacognitive model discriminates.
- **Haiku result:** 0.90 discrimination — accepts 90% of true corrections, resists 100% of false ones.

#### Benchmark 7: Difficulty Self-Ranking (t10) — METACOGNITIVE MONITORING
- **Competition target:** Prospective error prediction
- **Metric:** Spearman ρ = 0.3398
- **Items:** 5 sets × 30 questions (shuffled to remove positional bias)
- **What it measures:** Model ranks 30 questions from "hardest for me" to "easiest for me" before answering. Spearman correlation with actual correctness.
- **Why it's special:** Tests continuous difficulty perception (not binary correct/wrong). Shuffling prevents the model from gaming via question position.
- **Haiku result:** ρ=+0.34 (positive = correct direction, moderate strength).

#### Benchmark 8: Multi-Turn Belief Revision (t46) — BAYESIAN UPDATING
- **Competition target:** Metacognitive control
- **Metric:** Trajectory correlation = 0.8870
- **Items:** 8 scenarios × 5-6 evidence rounds each
- **What it measures:** Over multiple turns, model receives supporting/opposing/neutral evidence and updates its confidence. Trajectory compared to expert-calibrated reference.
- **Why it's special:** Tests dynamic metacognition — not just one-shot assessment, but whether the model integrates evidence rationally over time.
- **Haiku result:** r=0.89, direction accuracy=100% (always adjusts in correct direction).

---

## 3. Pre-Submission Checklist

### Critical (Must do)
- [ ] **Audit Opus-generated datasets for factual accuracy** — Sample 50 items from t01, t09, t10, t11, t48 and verify correct_answer fields manually. Fix any errors found.
- [ ] **Run evaluation on at least 2 models** — Haiku 4.5 + Sonnet 4.6 (or Opus 4.6) to show the benchmarks discriminate between model capabilities.
- [ ] **Write the submission paper/notebook** — Clear narrative, methodology, results, analysis.
- [ ] **Add statistical significance tests** — Bootstrap confidence intervals on primary metrics.

### Important (Should do)
- [ ] **Clean up the t19 story** — The null result (gap=0) is interesting but needs framing: "We found no self-other bias in math verification, suggesting models may not exhibit ownership bias in this domain." This is a finding, not a failure.
- [ ] **Add reliability diagrams** — Visual calibration plots for t01 and t48 (judges love figures).
- [ ] **Run t03 and t07 with Opus-generated data** — Currently using v1 programmatic datasets.
- [ ] **Add cross-model comparison table** — Show how metrics change across Haiku/Sonnet/Opus.

### Nice to Have
- [ ] **Package as pip-installable** — `pip install metacognition-benchmarks`
- [ ] **Add a leaderboard script** — Auto-generates comparison table across models
- [ ] **Publish datasets on HuggingFace** — For reproducibility

---

## 4. Submission Structure

### If submitting as a paper/notebook:

```
1. Introduction (0.5 page)
   - Metacognition in AI — the gap between "what models know" and "what they know they know"
   - Our contribution: 5-pillar taxonomy + 32 benchmarks + empirical validation

2. Framework (1 page)
   - The 5 pillars (table + 1 paragraph each)
   - Design principles: behavioral measures, controlled experiments, multi-turn

3. Selected Benchmarks (2-3 pages, ~0.3 page each)
   - For each of the 8 selected benchmarks:
     - Motivation (1-2 sentences)
     - Method (prompt design, scoring)
     - Results on Haiku 4.5 (metric + key sub-metrics)
     - What it reveals about the model

4. Cross-Benchmark Analysis (0.5 page)
   - Pillar-level scores
   - Where Haiku excels (metacognitive control) vs struggles (calibration)
   - The headline number: t48 AUROC

5. Discussion (0.5 page)
   - Null results (t19) and what they mean
   - Limitations (LLM-as-judge circularity, dataset quality)
   - Future work (more models, harder datasets)

6. Reproducibility
   - pip install + one command to run
   - All datasets included
   - MIT license
```

### Key figures to include:
1. **Pillar radar chart** — 5 axes showing Haiku's metacognition profile
2. **Abstention ROC curve** (t48) — visual of accuracy vs coverage tradeoff
3. **Calibration reliability diagram** (t01) — predicted vs actual accuracy by confidence bin
4. **Belief revision trajectories** (t46) — model vs expert confidence over evidence rounds
5. **v1 vs v2 comparison table** — shows iterative hardening process

---

## 5. Competitive Positioning

### Our differentiators vs likely competitors:
| What others will do | What we do differently |
|--------------------|-----------------------|
| Single calibration benchmark | 5-pillar taxonomy with 32 benchmarks |
| Ask for 0-100 confidence | Behavioral economics game (t11), multi-turn dynamics (t46) |
| Test on one model | Validated framework with parallel workers, tested on Haiku |
| Static question set | Opus-generated hard datasets + iterative difficulty calibration |
| Single metric | Headline AUROC + per-pillar breakdown |
| Factual QA only | Controlled experiments (t19), sycophancy testing (t22), belief revision (t46) |

### Our strongest selling points (in order):
1. **Comprehensiveness** — 5 pillars, 32 benchmarks, 3,808 items
2. **Experimental design novelty** — t19 (attribution control), t11 (economic game), t22 (sycophancy probe)
3. **Empirical validation** — Real results on Haiku 4.5, including null results
4. **Practical framework** — One pip install, one command, parallel execution, JSON results
5. **Iterative rigor** — v1 too easy → diagnosed → v2 with Opus-hardened datasets → re-evaluated

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Opus-generated answers contain factual errors | Audit 50+ items manually before submission |
| t19 null result looks like benchmark failure | Frame as genuine scientific finding about model cognition |
| LLM-as-judge circularity | Acknowledge in limitations; show it doesn't affect primary metrics (most use exact match + aliases first) |
| Other teams have more novel single benchmarks | Emphasize our framework's breadth + the fact that each pillar has 5-8 benchmarks |
| Judges can't run it (no Bedrock access) | Provide pre-computed results; optionally add OpenAI/local model backends |

---

## 7. Timeline

| Day | Action |
|-----|--------|
| Day 1 | Audit Opus datasets (50 items from each of t01, t09, t10, t11, t48) |
| Day 1 | Run evaluation on Sonnet 4.6 for cross-model comparison |
| Day 2 | Write submission paper/notebook (sections 1-4) |
| Day 2 | Generate figures (radar chart, ROC curve, calibration diagram, trajectories) |
| Day 3 | Write sections 5-6, polish, internal review |
| Day 3 | Package repository (README, requirements, license) |
| Day 4 | Final review, submit |
