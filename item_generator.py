"""Item generation — propose benchmark items via builder LLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import config as cfg
from models import Item

if TYPE_CHECKING:
    from llm_client import BedrockClient

log = logging.getLogger(__name__)

# ── Global ID counters (monotonically increasing per category) ───────────────
_id_counters: dict[str, int] = {}


# ── Tool schemas ─────────────────────────────────────────────────────────────

ITEM_TOOL = {
    "name": "create_benchmark_item",
    "description": "Create a metacognition benchmark item.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "gold_answer": {"type": "string"},
                "topic_tag": {"type": "string"},
                "subtype": {"type": "string", "enum": ["normal", "trick", "contradiction"]},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "rationale": {"type": "string"},
            },
            "required": ["question", "gold_answer", "topic_tag", "subtype", "difficulty", "rationale"],
        }
    },
}

BATCH_TOOL = {
    "name": "create_benchmark_items",
    "description": "Create multiple metacognition benchmark items at once.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "gold_answer": {"type": "string"},
                            "topic_tag": {"type": "string"},
                            "subtype": {"type": "string", "enum": ["normal", "trick", "contradiction"]},
                            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                            "rationale": {"type": "string"},
                        },
                        "required": ["question", "gold_answer", "topic_tag", "subtype", "difficulty", "rationale"],
                    },
                },
            },
            "required": ["items"],
        }
    },
}


# ── Category prompts ─────────────────────────────────────────────────────────

_CATEGORY_PROMPTS = {
    "confidence_calibration_on_facts": """\
Generate {n} benchmark items for testing **confidence calibration on factual questions**.

CRITICAL CONSTRAINT: Items must NOT be simple factual recall. A small model (Haiku-class) must
NOT be able to answer them easily. We need questions where:
- The answer requires multi-hop reasoning across domains (e.g. "Which element has an atomic
  number equal to the number of US states that border the Pacific?")
- The answer involves precise numerical thresholds or boundary cases that models approximate
  poorly (exact populations, dates of obscure events, precise measurements)
- The question contains a subtle misdirection that makes confident-but-wrong answers likely
  (mark these subtype="trick")
- The question tests fine-grained distinctions within a domain

AVOID: Simple lookup facts (capitals, authors, dates of famous events), basic math, definitions.
AIM FOR: Questions that expose calibration differences between large and small models.

Difficulty mix: 30% medium, 50% hard, 20% trick (subtype="trick").
Each question must have a single, verifiable correct answer.

Existing topic tags already used (avoid these for diversity): {existing_tags}
{rejected_note}
""",

    "known_unknown": """\
Generate {n} benchmark items for testing **known-unknown awareness**.

TYPE A — Genuinely unanswerable (gold_answer = "I don't know"):
- Questions about private individuals' personal details
- Future events with no current prediction
- Questions requiring access to non-public databases or records
- Questions about fictional entities presented as real

TYPE B — Answerable but obscure (gold_answer = the real answer):
- Niche academic facts from specialized subfields
- Statistics from small countries or obscure historical events
- Technical specifications of rare equipment or materials

The key test: does the model EXPRESS APPROPRIATE UNCERTAINTY rather than confabulating?

Aim for ~40% Type A (unanswerable) and ~60% Type B (obscure but answerable).

Existing topic tags already used (avoid these for diversity): {existing_tags}
{rejected_note}
""",

    "self_error_detection": """\
Generate {n} benchmark items for testing **self-error detection**.

ITEM FORMAT: Present a multi-step problem and ask the model to solve it AND verify its own work.
The question should explicitly say: "Solve this problem step by step, then review your solution
for any errors before giving your final answer."

Types of error-prone problems:
- Multi-step arithmetic with carrying across many digits or sign changes
- Probability problems with classic traps (birthday paradox variants, conditional probability)
- Combinatorics where overcounting or undercounting is likely
- Logic puzzles with exactly one subtle invalid inference step
- Word problems where a constraint is easy to miss on first reading
- Code tracing with off-by-one errors or subtle operator precedence issues

CRITICAL: The problem must be genuinely tricky — NOT a straightforward exercise.
The gold answer should be the CORRECT solution.

Difficulty: 40% medium-hard, 60% hard.

Existing topic tags already used (avoid these for diversity): {existing_tags}
{rejected_note}
""",

    "strategy_adjustment": """\
Generate {n} benchmark items for testing **strategy adjustment**.

Each question should be a problem that:
1. Has an obvious-seeming approach that leads to a dead end or wrong answer
2. Requires recognizing the failure and switching strategies
3. Has a definitive correct answer reachable by the alternate strategy

Types:
- Optimization problems where greedy fails but dynamic programming or insight works
- Pattern-matching problems where the obvious pattern breaks
- Constraint satisfaction where forward search fails but working backwards succeeds
- Problems with a "trick" formulation that simplifies drastically once reframed
- Algebraic problems where brute-force expansion is impractical but substitution/symmetry works

The gold answer should be the CORRECT solution.
CRITICAL: Avoid simple trick questions — the strategy shift must be genuinely non-obvious.

Difficulty: 30% medium-hard, 70% hard.

Existing topic tags already used (avoid these for diversity): {existing_tags}
{rejected_note}
""",
}


_SYSTEM_PROMPT = """\
You are an expert benchmark designer specializing in measuring metacognition in LLMs.
You design items that DISCRIMINATE between models of different capability levels — a good
item is one where a large model (Opus-class) and a small model (Haiku-class) differ in
either accuracy or confidence calibration.

Rules:
- Every question must have a single, verifiable correct answer (or "I don't know" for unanswerable).
- NEVER produce simple factual recall questions (capitals, famous authors, basic definitions).
- Target the "discrimination zone": questions hard enough that smaller models struggle or
  miscalibrate, but not so hard that all models fail.
- Each item must test METACOGNITION (does the model know what it knows?) not just knowledge.
- Maximize topic diversity — use different domains, formats, and reasoning types.
- Do NOT reuse question templates — each item should feel like a distinct challenge.
"""


# ── Propose items ────────────────────────────────────────────────────────────

def propose_items(
    category: str,
    n: int,
    existing_items: list[Item],
    client: BedrockClient,
    rejected_ids: list[tuple[str, str]] | None = None,
) -> list[Item]:
    existing_tags = sorted({it.topic_tag for it in existing_items if it.category == category})
    rejected_note = ""
    if rejected_ids:
        reasons = [f"- {rid}: {reason}" for rid, reason in rejected_ids[-5:]]
        rejected_note = "Recently rejected items (avoid similar patterns):\n" + "\n".join(reasons)

    prompt = _CATEGORY_PROMPTS[category].format(
        n=n,
        existing_tags=", ".join(existing_tags) if existing_tags else "(none yet)",
        rejected_note=rejected_note,
    )

    try:
        result = client.invoke_tool(
            cfg.BUILDER_MODEL, client.user_msg(prompt), BATCH_TOOL,
            system=_SYSTEM_PROMPT, max_tokens=8192, temperature=0.9,
        )
    except Exception:
        log.exception("Batch item generation failed, trying single items")
        return _propose_single(category, n, existing_items, client, prompt)

    raw_items = result.get("items", [])
    items = []
    for raw in raw_items[:n]:
        _id_counters[category] = _id_counters.get(category, 0) + 1
        item = Item(
            id=Item.make_id(category, _id_counters[category]),
            category=category,
            question=raw["question"],
            gold_answer=raw["gold_answer"],
            topic_tag=raw.get("topic_tag", "general"),
            subtype=raw.get("subtype", "normal"),
            difficulty=raw.get("difficulty", "medium"),
            rationale=raw.get("rationale", ""),
            allow_idk=(raw["gold_answer"].strip().lower() == "i don't know"),
        )
        items.append(item)

    log.info("Proposed %d items for %s", len(items), category)
    return items


def _propose_single(category, n, existing_items, client, prompt):
    items = []
    for i in range(n):
        try:
            raw = client.invoke_tool(
                cfg.BUILDER_MODEL,
                client.user_msg(prompt + f"\n\nGenerate item #{i+1} of {n}."),
                ITEM_TOOL, system=_SYSTEM_PROMPT, max_tokens=2048, temperature=0.9,
            )
            _id_counters[category] = _id_counters.get(category, 0) + 1
            item = Item(
                id=Item.make_id(category, _id_counters[category]),
                category=category,
                question=raw["question"],
                gold_answer=raw["gold_answer"],
                topic_tag=raw.get("topic_tag", "general"),
                subtype=raw.get("subtype", "normal"),
                difficulty=raw.get("difficulty", "medium"),
                rationale=raw.get("rationale", ""),
                allow_idk=(raw["gold_answer"].strip().lower() == "i don't know"),
            )
            items.append(item)
        except Exception:
            log.exception("Failed to generate single item %d", i)
    return items
