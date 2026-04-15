"""Configuration for metacognition benchmark builder."""

import os

# ── AWS Bedrock ──────────────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

BEDROCK_MODELS = {
    # Claude family
    "opus-4.6":   "us.anthropic.claude-opus-4-6-v1",
    "sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
    "haiku-4.5":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "opus-4.5":   "us.anthropic.claude-opus-4-5-20251101-v1:0",
    # Amazon Nova family
    "nova-micro":  "amazon.nova-micro-v1:0",
    "nova-lite":   "amazon.nova-lite-v1:0",
    "nova-pro":    "amazon.nova-pro-v1:0",
    # Google Gemma family
    "gemma-3-4b":  "google.gemma-3-4b-it",
    "gemma-3-12b": "google.gemma-3-12b-it",
    "gemma-3-27b": "google.gemma-3-27b-it",
}

# ── Roles ────────────────────────────────────────────────────────────────────
BUILDER_MODEL   = "sonnet-4.6"        # generates benchmark items
VALIDATOR_MODEL = "haiku-4.5"         # ambiguity kill-signal
JUDGE_MODEL     = "opus-4.6"          # LLM-as-a-judge scorer

# Reference models for ECE spread (used during search)
REF_MODELS = ["sonnet-4.6", "opus-4.6", "haiku-4.5", "opus-4.5"]

# Cross-model eval (used during final evaluation)
CROSS_MODELS = ["nova-micro", "nova-lite", "nova-pro", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"]

# ── Taxonomy (frozen) ────────────────────────────────────────────────────────
CATEGORIES = [
    "confidence_calibration_on_facts",
    "known_unknown",
    "self_error_detection",
    "strategy_adjustment",
]

CATEGORY_PREFIX = {
    "confidence_calibration_on_facts": "cal",
    "known_unknown":                   "ku",
    "self_error_detection":            "sed",
    "strategy_adjustment":             "sa",
}

# ── Search ───────────────────────────────────────────────────────────────────
BEAM_WIDTH              = 1
CANDIDATES_PER_EXPAND   = 10
ITEMS_PER_CATEGORY      = 75
TOTAL_ITEMS             = ITEMS_PER_CATEGORY * len(CATEGORIES)  # 300
MAX_STALE_LAYERS        = 2

# ── Hard constraints ─────────────────────────────────────────────────────────
MIN_ITEMS_PER_CATEGORY          = 15
MAX_TRICK_RATIO                 = 0.15
GLOBAL_NOVELTY_THRESHOLD        = 0.85
INTRA_NOVELTY_THRESHOLD         = 0.75
MIN_TOPIC_TAGS_PER_CATEGORY     = 5
METACOGNITION_ISOLATION_TAU     = 0.05

# ── ECE ──────────────────────────────────────────────────────────────────────
ECE_BINS = 10

# ── Kill-signal thresholds ───────────────────────────────────────────────────
SATURATION_CONF_THRESHOLD = 0.95

# ── Concurrency / rate limits ────────────────────────────────────────────────
MAX_WORKERS         = 20
REQUEST_TIMEOUT_SEC = 120
MAX_RETRIES         = 3
RETRY_BACKOFF_BASE  = 2.0

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
