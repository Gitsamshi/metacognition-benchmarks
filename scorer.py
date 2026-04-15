"""Frozen scorer — LLM-as-a-judge for all non-trivial comparisons."""

from __future__ import annotations

import logging
import re
import unicodedata

import config as cfg

log = logging.getLogger(__name__)


# ── Normalisation ────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def string_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


# ── LLM-as-a-Judge (primary scorer) ─────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict answer-equivalence judge for a benchmark.\n"
    "Given a GOLD answer and a PREDICTED answer, determine if the predicted "
    "answer is correct — i.e. it conveys the same factual content as the gold "
    "answer. Ignore differences in wording, verbosity, formatting, or extra "
    "explanation. Focus ONLY on whether the core factual claim matches.\n\n"
    "Special cases:\n"
    "- If the gold answer is 'I don't know' (or equivalent), the prediction is "
    "correct if and only if it also refuses to answer, says it cannot determine "
    "the answer, or expresses that the information is unavailable/unknowable.\n"
    "- If the gold is a number, accept reasonable rounding.\n"
    "- If the prediction contains the correct answer embedded in a longer "
    "explanation, that counts as correct."
)

_JUDGE_TOOL = {
    "name": "judge_equivalence",
    "description": "Report whether the predicted answer is equivalent to gold.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "equivalent": {
                    "type": "boolean",
                    "description": "true if the prediction is factually correct given the gold answer",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for your judgment",
                },
            },
            "required": ["equivalent", "reason"],
        }
    },
}


def llm_judge(prediction: str, gold: str, client) -> bool:
    """Use LLM judge to decide answer correctness. Primary scoring method."""
    prompt = (
        f"GOLD ANSWER: {gold}\n\n"
        f"PREDICTED ANSWER: {prediction}\n\n"
        f"Is the predicted answer correct (equivalent to the gold answer)?"
    )
    # Attempt 1: tool_use
    try:
        result = client.invoke_tool(
            cfg.JUDGE_MODEL,
            client.user_msg(prompt),
            _JUDGE_TOOL,
            system=_JUDGE_SYSTEM,
            max_tokens=512,
            temperature=0.0,
        )
        return bool(result.get("equivalent", False))
    except Exception:
        pass

    # Attempt 2: plain text
    try:
        text = client.invoke_text(
            cfg.JUDGE_MODEL,
            client.user_msg(prompt),
            system=_JUDGE_SYSTEM,
            max_tokens=512,
            temperature=0.0,
        )
        tl = text.lower()
        if '"equivalent": true' in tl or '"equivalent":true' in tl:
            return True
        if '"equivalent": false' in tl or '"equivalent":false' in tl:
            return False
        if tl.strip().startswith("yes") or "is correct" in tl or "are equivalent" in tl:
            return True
        if tl.strip().startswith("no") or "is incorrect" in tl or "not equivalent" in tl:
            return False
    except Exception:
        pass

    log.warning("LLM judge failed both attempts, falling back to string match")
    return string_match(prediction, gold)


# ── Combined scorer ──────────────────────────────────────────────────────────

def score(prediction: str, gold: str, client=None) -> bool:
    """Determine correctness. Always uses LLM judge when client is available."""
    # Fast path: exact string match is always correct
    if string_match(prediction, gold):
        return True

    # LLM judge (primary method)
    if client is not None:
        return llm_judge(prediction, gold, client)

    # No client — cannot score reliably
    return False
