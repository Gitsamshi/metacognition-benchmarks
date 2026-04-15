"""Kill-signal pipeline — ordered cheapest-first.

Stages:
  1. Saturation — all models correct with high conf
  2. Floor     — zero models correct
  3. Ambiguity — independent LLM disagrees on gold
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import config as cfg
from models import Item, RefRun, KillSignal

if TYPE_CHECKING:
    from llm_client import BedrockClient

log = logging.getLogger(__name__)


def check_saturation(runs: list[RefRun]) -> KillSignal:
    if not runs:
        return KillSignal("saturation", False, "no runs")
    all_correct = all(r.is_correct for r in runs)
    all_high_conf = all(r.stated_conf >= cfg.SATURATION_CONF_THRESHOLD for r in runs)
    triggered = all_correct and all_high_conf
    detail = ""
    if triggered:
        confs = [f"{r.model}={r.stated_conf:.2f}" for r in runs]
        detail = f"all correct, high conf: {', '.join(confs)}"
    return KillSignal("saturation", triggered, detail)


def check_floor(runs: list[RefRun]) -> KillSignal:
    if not runs:
        return KillSignal("floor", False, "no runs")
    n_correct = sum(r.is_correct for r in runs)
    if n_correct == 0:
        return KillSignal("floor", True, "all models wrong")
    return KillSignal("floor", False)


_AMBIGUITY_SYSTEM = (
    "You are an answer validator. Given a question and a proposed gold-standard "
    "answer, determine if the gold answer is CORRECT and UNAMBIGUOUS.\n"
    "Respond ONLY with JSON: {\"agree\": true/false, \"reason\": \"...\"}"
)

_AMBIGUITY_TOOL = {
    "name": "validate_gold",
    "description": "Validate whether the gold answer is correct and unambiguous.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "agree": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["agree"],
        }
    },
}


def check_ambiguity(item: Item, client: BedrockClient) -> KillSignal:
    prompt = (
        f"Question: {item.question}\n"
        f"Proposed gold answer: {item.gold_answer}\n\n"
        "Is this gold answer correct and unambiguous?"
    )
    try:
        result = client.invoke_tool(
            cfg.VALIDATOR_MODEL,
            client.user_msg(prompt),
            _AMBIGUITY_TOOL,
            system=_AMBIGUITY_SYSTEM,
            max_tokens=256,
            temperature=0.0,
        )
        agree = result.get("agree", True)
        reason = result.get("reason", "")
        if not agree:
            return KillSignal("ambiguity", True, f"validator disagrees: {reason}")
        return KillSignal("ambiguity", False)
    except Exception as exc:
        log.warning("Ambiguity check failed: %s", exc)
        return KillSignal("ambiguity", False, f"error: {exc}")


def run_kill_pipeline(item: Item, runs: list[RefRun], client: BedrockClient) -> list[KillSignal]:
    signals: list[KillSignal] = []

    sig = check_saturation(runs)
    signals.append(sig)
    if sig.triggered:
        log.info("Kill: %s — %s — %s", item.id, sig.signal_type, sig.detail)
        return signals

    sig = check_floor(runs)
    signals.append(sig)
    if sig.triggered:
        log.info("Kill: %s — %s — %s", item.id, sig.signal_type, sig.detail)
        return signals

    sig = check_ambiguity(item, client)
    signals.append(sig)
    if sig.triggered:
        log.info("Kill: %s — %s — %s", item.id, sig.signal_type, sig.detail)
        return signals

    return signals
