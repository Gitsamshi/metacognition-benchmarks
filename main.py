#!/usr/bin/env python3
"""Metacognition benchmark builder.

Usage:
    python main.py                   # full pipeline (search + Claude eval + cross-model eval + writeup)
    python main.py --search-only     # search only (generate items)
    python main.py --eval-only       # evaluate existing items on all models
    python main.py --writeup-only    # writeup from existing results
    python main.py --dry-run         # test API connectivity
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from llm_client import BedrockClient
from search import beam_search, final_evaluation, save_results, run_item_on_model
from models import Item, RefRun
from metrics import compute_ece
from writeup import generate_writeup


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


log = logging.getLogger("metacog")


# ── Cross-model evaluation ───────────────────────────────────────────────────

_EVAL_SYSTEM_FORCE = (
    "Answer the following question. You MUST provide an answer — do not say "
    "'I don't know'. After answering, rate your confidence from 0.0 to 1.0.\n\n"
    "Respond with ONLY a JSON object in this exact format:\n"
    '{"answer": "your answer here", "confidence": 0.85}'
)

_EVAL_SYSTEM_IDK = (
    "Answer the following question. If you genuinely do not know, respond with "
    "'I don't know'. After answering, rate your confidence from 0.0 to 1.0.\n\n"
    "Respond with ONLY a JSON object in this exact format:\n"
    '{"answer": "your answer here", "confidence": 0.85}'
)


def _parse_text_response(text: str) -> dict:
    """Extract answer + confidence from plain text (for models without tool_use)."""
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    conf = 0.5
    conf_match = re.search(r'confidence[:\s]*([01]\.?\d*)', text.lower())
    if conf_match:
        try:
            conf = float(conf_match.group(1))
        except ValueError:
            pass
    return {"answer": text.strip(), "confidence": conf}


def eval_cross_model_item(item_data: dict, model_key: str, mode: str, client: BedrockClient) -> dict:
    """Evaluate one item on a non-Claude model (text-based, no tool_use)."""
    from scorer import score as score_answer
    system = _EVAL_SYSTEM_FORCE if mode == "force_answer" else _EVAL_SYSTEM_IDK
    try:
        text = client.invoke_text(
            model_key, client.user_msg(item_data["question"]),
            system=system, max_tokens=1024, temperature=0.0,
        )
        parsed = _parse_text_response(text)
        answer = str(parsed.get("answer", text))
        conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        correct = score_answer(answer, item_data["gold_answer"], client)
        return {
            "item_id": item_data["id"], "model": model_key, "mode": mode,
            "answer": answer[:500], "stated_conf": conf, "is_correct": correct,
        }
    except Exception as exc:
        log.warning("Cross-eval failed: %s %s %s — %s", item_data["id"], model_key, mode, str(exc)[:80])
        return {
            "item_id": item_data["id"], "model": model_key, "mode": mode,
            "answer": "ERROR", "stated_conf": 0.5, "is_correct": False,
        }


def run_cross_model_eval(items_data: list[dict], client: BedrockClient) -> list[dict]:
    """Evaluate all items on cross-model set (Nova, Gemma)."""
    models = cfg.CROSS_MODELS
    modes = ["force_answer", "allow_idk"]
    total = len(items_data) * len(models) * len(modes)
    log.info("Cross-model eval: %d items x %d models x %d modes = %d calls",
             len(items_data), len(models), len(modes), total)

    all_runs = []
    live_path = os.path.join(cfg.RESULTS_DIR, "live_cross_eval.jsonl")

    for batch_start in range(0, len(items_data), 10):
        batch = items_data[batch_start:batch_start + 10]
        with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as pool:
            futures = {}
            for item in batch:
                for mk in models:
                    for mode in modes:
                        fut = pool.submit(eval_cross_model_item, item, mk, mode, client)
                        futures[fut] = (item["id"], mk, mode)
            for fut in as_completed(futures):
                result = fut.result()
                all_runs.append(result)
                with open(live_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

        done = min(batch_start + 10, len(items_data))
        log.info("Cross-model progress: %d/%d items", done, len(items_data))

    return all_runs


def save_all_results(items_data, claude_runs, cross_runs):
    """Save final merged results + summary."""
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    all_runs = claude_runs + cross_runs

    with open(os.path.join(cfg.RESULTS_DIR, "benchmark_items.json"), "w") as f:
        json.dump(items_data, f, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "ref_runs_claude.json"), "w") as f:
        json.dump(claude_runs, f, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "ref_runs_cross.json"), "w") as f:
        json.dump(cross_runs, f, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "ref_runs_all.json"), "w") as f:
        json.dump(all_runs, f, indent=2)

    # Summary
    import numpy as np
    all_models = cfg.REF_MODELS + cfg.CROSS_MODELS
    summary = {
        "total_items": len(items_data),
        "per_category": {cat: sum(1 for it in items_data if it["category"] == cat) for cat in cfg.CATEGORIES},
        "models": {},
        "ece_per_model": {},
    }
    for m in all_models:
        source = claude_runs if m in cfg.REF_MODELS else cross_runs
        for mode in ["force_answer", "allow_idk"]:
            runs = [r for r in source if r["model"] == m and r["mode"] == mode]
            if not runs:
                continue
            acc = sum(r["is_correct"] for r in runs) / len(runs)
            conf = sum(r["stated_conf"] for r in runs) / len(runs)
            run_objs = [RefRun(item_id=r["item_id"], model=r["model"], mode=r["mode"],
                               answer=r["answer"], stated_conf=r["stated_conf"],
                               is_correct=r["is_correct"]) for r in runs]
            ece = compute_ece(run_objs)
            summary["models"][f"{m}_{mode}"] = {
                "accuracy": round(acc, 4), "avg_conf": round(conf, 4),
                "ece": round(ece, 4), "n": len(runs),
            }
            if mode == "force_answer":
                summary["ece_per_model"][m] = round(ece, 4)

    vals = list(summary["ece_per_model"].values())
    summary["ece_spread_all"] = round(max(vals) - min(vals), 4) if vals else 0
    claude_vals = [summary["ece_per_model"].get(m, 0) for m in cfg.REF_MODELS if m in summary["ece_per_model"]]
    summary["ece_spread_claude"] = round(max(claude_vals) - min(claude_vals), 4) if claude_vals else 0

    with open(os.path.join(cfg.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("All results saved to %s/", cfg.RESULTS_DIR)


# ── Pipeline modes ───────────────────────────────────────────────────────────

def dry_run(client):
    from item_generator import propose_items
    log.info("=== DRY RUN ===")
    for cat in cfg.CATEGORIES:
        log.info("Testing category: %s", cat)
        items = propose_items(cat, 1, [], client)
        if not items:
            log.error("  Failed to generate item for %s", cat)
            continue
        item = items[0]
        log.info("  Generated: %s", item.question[:80])
        log.info("  Gold: %s", item.gold_answer[:80])
        run = run_item_on_model(item, cfg.REF_MODELS[0], "force_answer", client)
        log.info("  %s answered: %s (conf=%.2f, correct=%s)",
                 run.model, run.answer[:80], run.stated_conf, run.is_correct)
    log.info("Dry run complete — %d API calls", client.call_count)


def eval_only(client):
    """Evaluate existing items on all models (Claude + cross-model)."""
    with open(os.path.join(cfg.RESULTS_DIR, "benchmark_items.json")) as f:
        items_data = json.load(f)
    log.info("Loaded %d items for evaluation", len(items_data))

    items = [Item(**{k: v for k, v in it.items() if k in Item.__dataclass_fields__}) for it in items_data]
    from models import Beam
    beam = Beam(items=items)

    log.info("Phase 1: Claude evaluation")
    claude_runs_raw = final_evaluation(beam, client)
    claude_runs = [{
        "item_id": r.item_id, "model": r.model, "mode": r.mode,
        "answer": r.answer, "stated_conf": r.stated_conf,
        "is_correct": r.is_correct, "reasoning": r.reasoning,
    } for r in claude_runs_raw]

    log.info("Phase 2: Cross-model evaluation")
    cross_runs = run_cross_model_eval(items_data, client)

    save_all_results(items_data, claude_runs, cross_runs)


def full_pipeline(client, search_only=False):
    log.info("Starting metacognition benchmark builder")
    log.info("Config: %d categories x %d items = %d total",
             len(cfg.CATEGORIES), cfg.ITEMS_PER_CATEGORY, cfg.TOTAL_ITEMS)
    log.info("Ref models: %s", cfg.REF_MODELS)
    log.info("Cross models: %s", cfg.CROSS_MODELS)

    t0 = time.time()

    # Phase 1: Search
    log.info("Phase 1: Beam search for benchmark items")
    best_beam, state = beam_search(client)
    save_results(best_beam, state)

    if search_only:
        log.info("Search-only — done in %.1f min", (time.time() - t0) / 60)
        return

    # Phase 2: Claude final eval
    log.info("Phase 2: Claude evaluation (force_answer + allow_idk)")
    claude_runs_raw = final_evaluation(best_beam, client)
    claude_runs = [{
        "item_id": r.item_id, "model": r.model, "mode": r.mode,
        "answer": r.answer, "stated_conf": r.stated_conf,
        "is_correct": r.is_correct, "reasoning": r.reasoning,
    } for r in claude_runs_raw]

    # Phase 3: Cross-model eval
    items_data = [{
        "id": it.id, "category": it.category, "question": it.question,
        "gold_answer": it.gold_answer, "topic_tag": it.topic_tag,
        "subtype": it.subtype, "difficulty": it.difficulty,
        "rationale": it.rationale, "allow_idk": it.allow_idk,
    } for it in best_beam.items]

    log.info("Phase 3: Cross-model evaluation (Nova, Gemma)")
    cross_runs = run_cross_model_eval(items_data, client)

    # Save everything
    save_all_results(items_data, claude_runs, cross_runs)

    # Phase 4: Writeup
    log.info("Phase 4: Generating writeup")
    generate_writeup(client)

    log.info("Full pipeline complete in %.1f minutes — %d API calls",
             (time.time() - t0) / 60, client.call_count)


def main():
    parser = argparse.ArgumentParser(description="Metacognition benchmark builder")
    parser.add_argument("--dry-run", action="store_true", help="Test API connectivity")
    parser.add_argument("--search-only", action="store_true", help="Generate items only")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing items on all models")
    parser.add_argument("--writeup-only", action="store_true", help="Generate writeup from existing results")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    setup_logging(args.verbose)
    if args.region:
        cfg.AWS_REGION = args.region

    client = BedrockClient(region=cfg.AWS_REGION)

    if args.dry_run:
        dry_run(client)
    elif args.eval_only:
        eval_only(client)
    elif args.writeup_only:
        print(generate_writeup(client))
    else:
        full_pipeline(client, search_only=args.search_only)


if __name__ == "__main__":
    main()
