"""Generate Kaggle writeup from benchmark results."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import config as cfg

if TYPE_CHECKING:
    from llm_client import BedrockClient

log = logging.getLogger(__name__)

_WRITEUP_SYSTEM = """\
You are writing a Kaggle competition writeup for a metacognition benchmark submission.
The competition is "Measuring Progress Toward AGI — Cognitive Abilities" (Google DeepMind).

Write a clear, structured writeup with these sections:
1. **Problem** — What is metacognition and why measure it?
2. **Construction** — How the benchmark was built (search, LLM-as-a-judge scorer, kill signals, ECE spread).
3. **Dataset** — Summary of the 4 categories, item counts, difficulty distribution.
4. **Results** — ECE spread across models, per-category analysis, key findings.
5. **Insights** — Non-trivial observations about model metacognition.

Be concise, data-driven, and intellectually honest about limitations.
Use markdown formatting.
"""


def generate_writeup(client: BedrockClient) -> str:
    with open(os.path.join(cfg.RESULTS_DIR, "benchmark_items.json")) as f:
        items = json.load(f)
    with open(os.path.join(cfg.RESULTS_DIR, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(cfg.RESULTS_DIR, "ref_runs.json")) as f:
        runs = json.load(f)

    model_stats = {}
    for run in runs:
        m = run["model"]
        model_stats.setdefault(m, {"correct": 0, "total": 0, "confs": []})
        model_stats[m]["total"] += 1
        if run["is_correct"]:
            model_stats[m]["correct"] += 1
        model_stats[m]["confs"].append(run["stated_conf"])
    for m in model_stats:
        s = model_stats[m]
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0
        s["avg_conf"] = sum(s["confs"]) / len(s["confs"]) if s["confs"] else 0
        del s["confs"]

    prompt = (
        f"## Summary\n{json.dumps(summary, indent=2)}\n\n"
        f"## Model Performance\n{json.dumps(model_stats, indent=2)}\n\n"
        f"## Sample Items (first 2 per category)\n"
    )
    for cat in cfg.CATEGORIES:
        for it in [i for i in items if i["category"] == cat][:2]:
            prompt += f"- [{it['id']}] {it['question'][:120]}... (gold: {it['gold_answer'][:60]})\n"
        prompt += "\n"
    prompt += "\nWrite the full Kaggle writeup."

    writeup = client.invoke_text(
        cfg.BUILDER_MODEL, client.user_msg(prompt),
        system=_WRITEUP_SYSTEM, max_tokens=8192, temperature=0.3,
    )

    path = os.path.join(cfg.RESULTS_DIR, "writeup.md")
    with open(path, "w") as f:
        f.write(writeup)
    log.info("Writeup saved to %s", path)
    return writeup
