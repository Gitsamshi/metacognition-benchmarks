# Metacognition Benchmark Suite

A benchmark for measuring **metacognitive abilities** in frontier language models — calibration, known-unknown awareness, self-error detection, and strategy adjustment.

Built for the Kaggle "Measuring Progress Toward AGI — Cognitive Abilities" competition (Google DeepMind, Metacognition track).

## Results (300 items, 10 models)

### Force-Answer ECE (lower = better calibrated)

| Model | Accuracy | Confidence | ECE | Family |
|---|---|---|---|---|
| **opus-4.6** | 80.0% | 76.1% | **0.089** | Claude |
| haiku-4.5 | 70.3% | 68.0% | 0.122 | Claude |
| sonnet-4.6 | 79.7% | 78.1% | 0.130 | Claude |
| opus-4.5 | 81.3% | 70.6% | 0.141 | Claude |
| gemma-3-12b | 48.0% | 82.8% | 0.418 | Gemma |
| gemma-3-27b | 47.3% | 88.5% | 0.420 | Gemma |
| gemma-3-4b | 35.0% | 83.3% | 0.483 | Gemma |
| nova-pro | 40.0% | 91.7% | 0.517 | Nova |
| nova-lite | 32.0% | 91.6% | 0.630 | Nova |
| nova-micro | 27.3% | 93.0% | **0.670** | Nova |

**ECE spread**: 0.581 (cross-family), 0.052 (within Claude)

### Key findings

1. **Opus-4.6 is the best-calibrated model** (ECE 0.089) — its 76% confidence closely tracks 80% accuracy.
2. **Nova models are catastrophically overconfident** — 93% confidence on 27% accuracy (nova-micro).
3. **Model size does not guarantee calibration** — gemma-3-12b (12B) is better calibrated than gemma-3-27b (27B).
4. **Claude family clusters tightly** (ECE 0.089–0.141) vs open models (0.418–0.670), confirming the plan's prediction that intra-family ECE spread would be narrow.

## Benchmark design

### Categories (75 items each, 300 total)

| Category | What it measures | Item format |
|---|---|---|
| `confidence_calibration_on_facts` | Does stated confidence match accuracy? | Multi-hop factual questions requiring cross-domain reasoning |
| `known_unknown` | Can the model say "I don't know"? | Mix of genuinely unanswerable (40%) and obscure-but-answerable (60%) questions |
| `self_error_detection` | Does the model catch its own mistakes? | Error-prone multi-step problems with explicit self-review instruction |
| `strategy_adjustment` | Can the model pivot when stuck? | Problems where the obvious approach fails and a different strategy is needed |

### Construction pipeline

The benchmark is built via **greedy search** over item-category trees:

```
For each category (4 layers):
  1. PROPOSE  — Builder LLM (Sonnet 4.6) generates 10 candidate items
  2. FILTER   — Novelty check (TF-IDF cosine < 0.85 global, < 0.75 intra-category)
  3. EVALUATE — Run candidates on 4 Claude ref models in parallel
  4. KILL     — Pipeline rejects items that are:
                 - Saturated (all models correct + high confidence → no signal)
                 - Floor (all models wrong → likely buggy)
                 - Ambiguous (independent LLM disagrees on gold answer)
  5. SCORE    — Rank survivors by marginal ECE spread contribution
  6. ADD      — All survivors added to benchmark
  Repeat until 75 items per category
```

### Scorer: LLM-as-a-Judge

All correctness judgments use **Claude Opus 4.6 as judge** — not string matching. The judge prompt handles:
- Semantic equivalence (verbose gold vs concise prediction)
- IDK detection (refusals phrased as "cannot be determined", "future event", etc.)
- Numerical tolerance
- Embedded correct answers in longer explanations

String match is only used as an exact-match fast path.

### Kill signals

| Signal | Cost | Trigger |
|---|---|---|
| Saturation | Free (code) | All models correct with confidence ≥ 0.95 |
| Floor | Free (code) | Zero models correct |
| Ambiguity | 1 API call | Independent validator LLM disagrees on gold answer |

### Evaluation modes

Each item is evaluated in two modes:
- **force_answer** — model must answer, cannot say "I don't know"
- **allow_idk** — model may abstain

The gap between modes measures **metacognition isolation** — whether models benefit from being allowed to express uncertainty.

## Project structure

```
.
├── plan.md              # Search problem specification
├── config.py            # Model IDs, thresholds, beam parameters
├── models.py            # Data classes (Item, RefRun, Beam, etc.)
├── llm_client.py        # AWS Bedrock converse API client
├── scorer.py            # LLM-as-a-judge scorer (Opus 4.6)
├── metrics.py           # ECE, novelty, constraint checking
├── verifier.py          # Kill-signal pipeline
├── item_generator.py    # Category-specific item proposal via builder LLM
├── search.py            # Greedy search orchestrator + evaluation
├── writeup.py           # Kaggle writeup generator
├── main.py              # CLI entry point
├── requirements.txt
└── results/
    ├── benchmark_items.json    # 300 items (75 per category)
    ├── ref_runs_claude.json    # Claude family runs (300 items × 4 models × 2 modes)
    ├── ref_runs_cross.json     # Nova + Gemma runs (300 items × 6 models × 2 modes)
    ├── ref_runs_all.json       # All runs combined
    └── summary.json            # Metrics summary
```

## Usage

### Full pipeline (search + evaluate + writeup)

```bash
pip install -r requirements.txt
python main.py
```

This runs all 4 phases:
1. **Search** — Generate 300 items via greedy beam search (~3-4 hours)
2. **Claude eval** — Evaluate on opus-4.6, sonnet-4.6, haiku-4.5, opus-4.5 (~1 hour)
3. **Cross-model eval** — Evaluate on Nova micro/lite/pro, Gemma 4b/12b/27b (~30 min)
4. **Writeup** — Generate Kaggle writeup from results

### Other modes

```bash
python main.py --search-only     # Phase 1 only
python main.py --eval-only       # Phases 2+3 on existing items
python main.py --writeup-only    # Phase 4 from existing results
python main.py --dry-run         # Test API connectivity (1 item per category)
```

### Live monitoring

During the search phase, accepted items are written in real-time to:
```bash
tail -f results/live_items.jsonl | python3 -m json.tool
```

## Models evaluated

### Claude family (via AWS Bedrock)
| Model | Bedrock ID |
|---|---|
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` |
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude Opus 4.5 | `us.anthropic.claude-opus-4-5-20251101-v1:0` |

### Amazon Nova family
| Model | Bedrock ID |
|---|---|
| Nova Micro | `amazon.nova-micro-v1:0` |
| Nova Lite | `amazon.nova-lite-v1:0` |
| Nova Pro | `amazon.nova-pro-v1:0` |

### Google Gemma family
| Model | Bedrock ID |
|---|---|
| Gemma 3 4B | `google.gemma-3-4b-it` |
| Gemma 3 12B | `google.gemma-3-12b-it` |
| Gemma 3 27B | `google.gemma-3-27b-it` |

## Metrics

- **ECE (Expected Calibration Error)**: Measures how well a model's stated confidence matches its actual accuracy across 10 bins. Lower is better.
- **ECE spread**: `max(ECE) - min(ECE)` across models. Higher spread means the benchmark better discriminates between models.
- **Metacognition isolation gap**: `force_answer_acc - allow_idk_acc`. Negative gap means models benefit from being allowed to abstain.

## Requirements

- Python 3.10+
- AWS account with Bedrock access (Claude, Nova, Gemma models)
- IAM role with `bedrock:InvokeModel` permissions
- `boto3`, `numpy`, `scikit-learn`
