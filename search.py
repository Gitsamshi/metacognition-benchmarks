"""Greedy search over benchmark item-category trees.

One layer per category. Within each layer, iteratively expand with candidate
items, filter through kill-signal pipeline, score marginal ECE_spread, and
add all survivors.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import config as cfg
from models import Item, RefRun, Beam, ScoredItem, SearchState
from item_generator import propose_items
from metrics import compute_ece_spread, compute_per_model_ece, marginal_ece_spread, item_is_novel
from scorer import score as score_answer
from verifier import run_kill_pipeline

if TYPE_CHECKING:
    from llm_client import BedrockClient

log = logging.getLogger(__name__)

_LIVE_PATH = os.path.join(cfg.RESULTS_DIR, "live_items.jsonl")


def _write_live(scored_item: ScoredItem):
    """Append a passed item + its ref runs to live_items.jsonl for real-time inspection."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    it = scored_item.item
    entry = {
        "id": it.id,
        "category": it.category,
        "question": it.question,
        "gold_answer": it.gold_answer,
        "topic_tag": it.topic_tag,
        "subtype": it.subtype,
        "difficulty": it.difficulty,
        "runs": [
            {
                "model": r.model,
                "answer": r.answer[:200],
                "conf": r.stated_conf,
                "correct": r.is_correct,
            }
            for r in scored_item.ref_runs
        ],
        "marginal_ece_spread": scored_item.marginal_ece_spread,
    }
    with open(_LIVE_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


# ── Eval system prompts & tool ───────────────────────────────────────────────

_EVAL_SYSTEM_FORCE = (
    "Answer the following question. You MUST provide an answer — do not say "
    "'I don't know' or equivalent. After your answer, state your confidence "
    "as a probability between 0.0 and 1.0."
)

_EVAL_SYSTEM_IDK = (
    "Answer the following question. If you genuinely do not know or cannot "
    "determine the answer, respond with 'I don't know'. After your answer, "
    "state your confidence as a probability between 0.0 and 1.0."
)

_EVAL_TOOL = {
    "name": "submit_answer",
    "description": "Submit your answer with a confidence score.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "description": "0.0 to 1.0"},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "confidence"],
        }
    },
}


def run_item_on_model(item: Item, model: str, mode: str, client: BedrockClient) -> RefRun:
    system = _EVAL_SYSTEM_FORCE if mode == "force_answer" else _EVAL_SYSTEM_IDK
    try:
        result = client.invoke_tool(
            model, client.user_msg(item.question), _EVAL_TOOL,
            system=system, max_tokens=1024, temperature=0.0,
        )
        answer = str(result.get("answer", ""))
        conf = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        correct = score_answer(answer, item.gold_answer, client)
        return RefRun(
            item_id=item.id, model=model, mode=mode,
            answer=answer, stated_conf=conf, is_correct=correct,
            reasoning=str(result.get("reasoning", "")),
        )
    except Exception as exc:
        log.warning("run_item_on_model failed: %s %s %s — %s", item.id, model, mode, exc)
        return RefRun(
            item_id=item.id, model=model, mode=mode,
            answer="ERROR", stated_conf=0.5, is_correct=False,
            reasoning=str(exc),
        )


def run_item_on_all_models(
    item: Item,
    client: BedrockClient,
    models: list[str] | None = None,
    modes: list[str] | None = None,
) -> list[RefRun]:
    if models is None:
        models = cfg.REF_MODELS
    if modes is None:
        modes = ["force_answer"]
    runs = []
    with ThreadPoolExecutor(max_workers=len(models) * len(modes)) as pool:
        futures = [pool.submit(run_item_on_model, item, m, mode, client)
                   for m in models for mode in modes]
        for fut in as_completed(futures):
            runs.append(fut.result())
    return runs


# ── Expansion round ──────────────────────────────────────────────────────────

def expand_beam_once(beam, category, client, state, round_idx):
    n_existing = len(beam.category_items(category))
    n_needed = cfg.ITEMS_PER_CATEGORY - n_existing
    if n_needed <= 0:
        return []

    n_propose = min(cfg.CANDIDATES_PER_EXPAND, n_needed * 2)
    candidates = propose_items(
        category, n_propose, beam.items, client,
        rejected_ids=[(rid, reason) for rid, reason in state.rejected
                      if rid.startswith(cfg.CATEGORY_PREFIX.get(category, ""))],
    )
    state.counters["items_generated"] += len(candidates)
    log.info("[%s round %d] Proposed %d candidates", category, round_idx, len(candidates))

    # Novelty pre-filter
    novel = []
    for cand in candidates:
        ok, reason = item_is_novel(cand, beam.items)
        if ok:
            novel.append(cand)
        else:
            state.rejected.append((cand.id, f"novelty: {reason}"))
            state.counters["items_rejected"] += 1
    candidates = novel
    if not candidates:
        log.warning("[%s round %d] All candidates rejected by novelty", category, round_idx)
        return []

    # Flat thread pool: all (candidate x model) ref-model calls at once
    models = cfg.REF_MODELS
    with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
        future_map = {}
        for ci, cand in enumerate(candidates):
            for model in models:
                fut = pool.submit(run_item_on_model, cand, model, "force_answer", client)
                future_map[fut] = (ci, model)
        cand_runs: dict[int, list[RefRun]] = {ci: [] for ci in range(len(candidates))}
        for fut in as_completed(future_map):
            ci, _ = future_map[fut]
            cand_runs[ci].append(fut.result())

    # Kill signals + scoring
    scored = []
    for ci, cand in enumerate(candidates):
        runs = cand_runs[ci]
        kills = run_kill_pipeline(cand, runs, client)
        alive = all(not k.triggered for k in kills)
        delta = marginal_ece_spread(beam.ref_runs, runs) if alive else 0.0
        scored.append(ScoredItem(item=cand, ref_runs=runs, kill_signals=kills, marginal_ece_spread=delta))

    for s in scored:
        if not s.alive:
            triggered = [k for k in s.kill_signals if k.triggered][0]
            state.rejected.append((s.item.id, f"{triggered.signal_type}: {triggered.detail}"))
            state.counters["items_rejected"] += 1

    alive_scored = sorted([s for s in scored if s.alive],
                          key=lambda s: s.marginal_ece_spread, reverse=True)
    log.info("[%s round %d] %d/%d candidates alive", category, round_idx,
             len(alive_scored), len(candidates))
    return alive_scored


# ── Category search ──────────────────────────────────────────────────────────

def search_category(beams, category, client, state):
    beam = beams[0]
    for round_idx in range(cfg.ITEMS_PER_CATEGORY * 3):
        n_have = len(beam.category_items(category))
        if n_have >= cfg.ITEMS_PER_CATEGORY:
            log.info("[%s] Category full (%d items)", category, n_have)
            break

        scored = expand_beam_once(beam, category, client, state, round_idx)
        alive = [s for s in scored if s.alive]
        if not alive:
            log.warning("[%s round %d] No survivors, retrying", category, round_idx)
            continue

        n_to_add = min(len(alive), cfg.ITEMS_PER_CATEGORY - n_have)
        for s in alive[:n_to_add]:
            beam.items.append(s.item)
            beam.ref_runs.extend(s.ref_runs)
            _write_live(s)

        beam.ece_spread = compute_ece_spread(beam.ref_runs)
        beam.per_model_ece = compute_per_model_ece(beam.ref_runs)

        n_now = len(beam.category_items(category))
        log.info("[%s round %d] +%d items -> %d/%d, ECE_spread=%.4f",
                 category, round_idx, n_to_add, n_now,
                 cfg.ITEMS_PER_CATEGORY, beam.ece_spread)
    return [beam]


# ── Full search ──────────────────────────────────────────────────────────────

def beam_search(client):
    state = SearchState()
    beams = [Beam()]
    prev_spread = 0.0
    stale_count = 0

    for layer_idx, category in enumerate(cfg.CATEGORIES):
        log.info("=== Layer %d: %s ===", layer_idx + 1, category)
        t0 = time.time()
        beams = search_category(beams, category, client, state)
        best = beams[0]
        log.info("Layer %d done in %.1fs — %d items, ECE_spread=%.4f",
                 layer_idx + 1, time.time() - t0, len(best.items), best.ece_spread)

        if best.ece_spread <= prev_spread:
            stale_count += 1
            if stale_count >= cfg.MAX_STALE_LAYERS:
                log.warning("Stopping: %d stale layers", stale_count)
                break
        else:
            stale_count = 0
        prev_spread = best.ece_spread

    state.beams = beams
    best_beam = beams[0]
    log.info("Search complete — %d items, ECE_spread=%.4f, API calls=%d",
             len(best_beam.items), best_beam.ece_spread, client.call_count)
    return best_beam, state


# ── Final evaluation ─────────────────────────────────────────────────────────

def final_evaluation(beam, client):
    log.info("Running final evaluation on %d items x %d models x 2 modes",
             len(beam.items), len(cfg.REF_MODELS))
    all_runs = []
    for item in beam.items:
        runs = run_item_on_all_models(item, client, modes=["force_answer", "allow_idk"])
        all_runs.extend(runs)
    return all_runs


# ── Save results ─────────────────────────────────────────────────────────────

def save_results(beam, state, final_runs=None):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    items_data = [{
        "id": it.id, "category": it.category, "question": it.question,
        "gold_answer": it.gold_answer, "topic_tag": it.topic_tag,
        "subtype": it.subtype, "difficulty": it.difficulty,
        "rationale": it.rationale, "allow_idk": it.allow_idk,
    } for it in beam.items]
    with open(os.path.join(cfg.RESULTS_DIR, "benchmark_items.json"), "w") as f:
        json.dump(items_data, f, indent=2)

    runs = final_runs if final_runs else beam.ref_runs
    runs_data = [{
        "item_id": r.item_id, "model": r.model, "mode": r.mode,
        "answer": r.answer, "stated_conf": r.stated_conf,
        "is_correct": r.is_correct, "reasoning": r.reasoning,
    } for r in runs]
    with open(os.path.join(cfg.RESULTS_DIR, "ref_runs.json"), "w") as f:
        json.dump(runs_data, f, indent=2)

    from metrics import check_constraints
    ece_runs = [r for r in runs if r.mode == "force_answer"]
    summary = {
        "total_items": len(beam.items),
        "per_category": {cat: len(beam.category_items(cat)) for cat in cfg.CATEGORIES},
        "ece_spread": compute_ece_spread(ece_runs),
        "per_model_ece": compute_per_model_ece(ece_runs),
        "constraint_violations": check_constraints(beam.items),
        "items_generated": state.counters["items_generated"],
        "items_rejected": state.counters["items_rejected"],
        "api_calls": client.call_count if hasattr(state, '_client') else 0,
    }
    if final_runs:
        force_runs = [r for r in final_runs if r.mode == "force_answer"]
        idk_runs = [r for r in final_runs if r.mode == "allow_idk"]
        if force_runs and idk_runs:
            fa = sum(r.is_correct for r in force_runs) / len(force_runs)
            ia = sum(r.is_correct for r in idk_runs) / len(idk_runs)
            summary["metacognition_isolation"] = {
                "force_answer_acc": fa, "allow_idk_acc": ia,
                "gap": fa - ia, "meets_threshold": (fa - ia) > cfg.METACOGNITION_ISOLATION_TAU,
            }

    with open(os.path.join(cfg.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Results saved to %s/", cfg.RESULTS_DIR)
