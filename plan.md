# Search Problem Spec — Metacognition Benchmark Builder

**Source task**: Kaggle "Measuring Progress Toward AGI — Cognitive Abilities" (Google DeepMind, $200K, DDL 2026-04-16), Metacognition track.

**Reframe**: Instead of a model searching for answers, the **builder agent** (LLM) searches over possible benchmark designs. The goal is a submitted Personal Benchmark + Writeup that robustly measures metacognition (calibration, known-unknown, self-error-detection, strategy adjustment) in frontier models.

---

## §1. Problem

- **Artifact**:
  - `(a)` Personal Benchmark resource — items + frozen LLM-as-a-judge scorer + reference-model runs
  - `(b)` Kaggle writeup (Problem / Construction / Dataset / Results / Insights)
- **Goal predicate** (flips false → true when done):
  - writeup submitted to Kaggle ∧
  - on ≥2 frontier models, `ECE_spread > 0` with significance ∧
  - writeup contains ≥1 non-trivial insight
- **Optimization target**: internal, verifier-computable quality metrics — *not* the unobservable leaderboard score.

## §2. State

```
S = {
  items: {category → [Item]},              # grouped by fixed 4-cat taxonomy
  ref_runs: {model → [(item_id, answer, stated_conf)]},  # 1-shot each, cached
  metrics: {ECE_spread, per_category_ECE, coverage, novelty_matrix},
  hypotheses: [H],                         # "X-type items separate A from B"
  rejected: [(item, reason)],              # prevent re-proposal
  budget: {wallclock_remaining}
}
```

**Scorer is frozen at §0** (LLM-as-a-judge, not string match) and is not part of mutable state (reduces reward-hacking surface, at the cost of inflexibility).

**Taxonomy (frozen at §0)**:
1. `confidence_calibration_on_facts`
2. `known_unknown`
3. `self_error_detection`
4. `strategy_adjustment`

## §3. Action space `A`

Eight actions, all tool-call granularity (not token-level):

| Action | Description |
|---|---|
| `propose_item(category, seed)` | Generate a new candidate item (question + gold + assertion) |
| `mutate_item(item, op)` | Perturb an existing item (swap entity, add trap, shift difficulty) |
| `search_external(query)` | Web / arxiv / existing benchmarks for inspiration (**diversity > purity**, contamination handled by kill-signal) |
| `run_ref_models(item, models)` | Run item on reference models, get `{answer, stated_conf}` |
| `score_item(item)` | Compute marginal contribution to ECE_spread |
| `propose_hypothesis(h)` | Add an unverified claim to queue |
| `reject_item(item, reason)` | Mark rejected (prevents re-proposal) |
| `finalize_bench(subset)` | **Terminal, irreversible** — freeze benchmark |

## §4. Transition `T`

| Action | Type | Notes |
|---|---|---|
| `propose_item`, `mutate_item`, `propose_hypothesis` | **Stochastic** | Primary search leverage |
| `search_external` | Stochastic | Plus external noise |
| `run_ref_models` | Stochastic | **1-shot sampling** (no empirical-accuracy distribution) |
| `score_item`, `reject_item`, `finalize_bench` | Deterministic | Pure code |

## §5. Reward / Verifier `R`

### Kill-signal pipeline (ordered cheapest-first)

1. **Saturation** — all ref models correct with stated_conf ≈ 1 → no signal
2. **Floor** — all ref models wrong → likely buggy or no signal
3. **MMLU-proxy** — a closed-book small-model baseline answers correctly → tests knowledge, not metacognition
4. **Ambiguity** — independent second LLM disagrees on gold → item ambiguous
5. **Contamination** — paraphrase of item found via `search_external` → potentially in training data

### Main reward

```
R(bench) = ECE_spread(bench) = max_m ECE(m) − min_m ECE(m)
           over m ∈ {Sonnet-4.6, Opus-4.6, Haiku-4.5, Opus-4.5}
```

Item-level marginal reward (for beam ranking):
`ΔR(bench ∪ {item}) − R(bench)`

### Hard constraints (violation → −∞)

- Each category ≥ 15 items; trick/contradiction subtype ≤ 15% of total
- **Global novelty**: any item-pair cosine similarity < 0.85
- **Intra-category novelty**: any item-pair cosine < 0.75 (primary defense against template-gaming)
- Each category covers ≥ 5 distinct `topic_tag` values
- **Metacognition isolation**: `avg(force_answer_acc − allow_idk_acc) > τ`

### Grounding

- **Gold labels**: produced by `propose_item` (builder), validated by independent second LLM (ambiguity kill-signal)
- **stated_conf**: structured JSON field in ref-model output
- **actual_accuracy**: **LLM-as-a-judge** (Haiku-4.5) is the sole scorer for all non-trivial comparisons; exact string match used only as a fast-path when normalized strings are identical. No substring heuristics or regex matching — the judge handles IDK equivalence, numerical rounding, and verbose vs concise answer matching.
- **IDK items**: gold = "I don't know"; constructed to be *verifiably unanswerable* (e.g. future-dated events, properly-redacted specifics)

## §6. Search topology — Beam / Tree-of-Thoughts

**Why Beam over MCTS**: LLM-as-a-judge scorer is frozen → deep search would overfit the static reward. Beam's bounded depth is natural regularization.

| Parameter | Value |
|---|---|
| Layers | 4 (one per category, in taxonomy order) |
| Candidates per expansion (k) | 10 |
| Beam width (b) | 5 |
| Per-layer ranking | Marginal ΔECE_spread after kill-signal filter |
| Termination | All categories hit quota (25 items × 4 = 100) OR 2 consecutive layers with no ECE_spread gain |
| Parallelism | Partial — categories run independently, items within a category sequential |
| Budget | 8h search + 4h final ref_runs + 4h writeup (≈ 16h of 48h DDL) |

**Budget justification**: 500 rpm × 4 Claude models ≈ 2000 req/min; 100 items × 5 models = 500 requests ≈ 15s per full benchmark scoring. Verifier is cheap enough to treat as dense signal.

## §7. Failure modes

### Primary defense (chosen)

**#2 Category-quota gaming** (e.g. all 25 `known_unknown` items follow the same "date-based unanswerable" template):
- Intra-category novelty threshold 0.75 (strict)
- Required `topic_tag` diversity: ≥5 distinct tags per category
- Enforced at `score_item` time, not post-hoc

### Known risks (acknowledged, not actively defended)

- **Scorer overfit**: 4 ref models fixed → held-out model at evaluation may behave differently. Mitigation: diversify ref set (see "Weakest link" below).
- **stated_conf structural failure**: 1-shot sampling + JSON output → occasional field missing or out-of-range, direct ECE noise.
- **Contamination leak**: paraphrase detection has recall ceiling; semantically-equivalent restatements may slip through.

---

## Honest assessment

**Weakest link**: §5 reward depends on a 4-model reference set that is *intra-Claude-family* (all four models are Claude variants). If these four models are near-identical in metacognition behavior, `ECE_spread` collapses to noise and the entire search signal dies.

**Cheapest fix if time allows**: expand the reference set to at least one cross-family model (Gemini or GPT). This both:
1. Broadens `ECE_spread` dynamic range
2. Mitigates held-out generalization risk (the judge's evaluation model is almost certainly not a Claude variant)

**Secondary weakness**: 1-shot `run_ref_models` sampling. If budget allows k=3 resampling on the final ≤100 items (not during search), the calibration signal tightens ~√3× at 3× cost — still trivial under 500 rpm.

---

## TL;DR

Turn "build metacognition benchmark" into: **Beam search over item-category trees, driven by frozen LLM-as-a-judge scorer measuring cross-model ECE spread, with cheap kill-signals early-rejecting saturated/floor/ambiguous/contaminated items, and template-gaming defended via intra-category novelty threshold.**
