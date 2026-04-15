"""ECE, novelty, and constraint metrics."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg
from models import Item, RefRun


def compute_ece(runs: list[RefRun], n_bins: int = cfg.ECE_BINS) -> float:
    if not runs:
        return 0.0
    confs = np.array([r.stated_conf for r in runs])
    accs = np.array([float(r.is_correct) for r in runs])
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs > lo) & (confs <= hi)
        if lo == 0.0:
            mask |= confs == 0.0
        count = mask.sum()
        if count == 0:
            continue
        ece += (count / len(runs)) * abs(accs[mask].mean() - confs[mask].mean())
    return float(ece)


def compute_per_model_ece(all_runs: list[RefRun], models: list[str] = cfg.REF_MODELS) -> dict[str, float]:
    return {m: compute_ece([r for r in all_runs if r.model == m and r.mode == "force_answer"]) for m in models}


def compute_ece_spread(all_runs: list[RefRun], models: list[str] = cfg.REF_MODELS) -> float:
    per = compute_per_model_ece(all_runs, models)
    if not per:
        return 0.0
    vals = list(per.values())
    return max(vals) - min(vals)


def _item_text(item: Item) -> str:
    return f"{item.question} {item.gold_answer} {item.topic_tag}"


def item_is_novel(candidate: Item, existing: list[Item]) -> tuple[bool, str]:
    if not existing:
        return True, ""
    texts = [_item_text(it) for it in existing] + [_item_text(candidate)]
    vec = TfidfVectorizer().fit_transform(texts)
    sim = cosine_similarity(vec[-1:], vec[:-1])[0]
    for idx, s in enumerate(sim):
        if s >= cfg.GLOBAL_NOVELTY_THRESHOLD:
            return False, f"global similarity {s:.3f} with {existing[idx].id}"
        if existing[idx].category == candidate.category and s >= cfg.INTRA_NOVELTY_THRESHOLD:
            return False, f"intra-category similarity {s:.3f} with {existing[idx].id}"
    return True, ""


def novelty_violations(items: list[Item]) -> list[tuple[str, str, float, str]]:
    if len(items) < 2:
        return []
    texts = [_item_text(it) for it in items]
    vec = TfidfVectorizer().fit_transform(texts)
    sim = cosine_similarity(vec)
    violations = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = float(sim[i, j])
            if s >= cfg.GLOBAL_NOVELTY_THRESHOLD:
                violations.append((items[i].id, items[j].id, s, "global"))
            elif items[i].category == items[j].category and s >= cfg.INTRA_NOVELTY_THRESHOLD:
                violations.append((items[i].id, items[j].id, s, "intra"))
    return violations


def check_constraints(items: list[Item]) -> list[str]:
    violations = []
    by_cat: dict[str, list[Item]] = {}
    for it in items:
        by_cat.setdefault(it.category, []).append(it)
    for cat in cfg.CATEGORIES:
        cat_items = by_cat.get(cat, [])
        if len(cat_items) < cfg.MIN_ITEMS_PER_CATEGORY:
            violations.append(f"{cat}: only {len(cat_items)} items (need >={cfg.MIN_ITEMS_PER_CATEGORY})")
        tags = {it.topic_tag for it in cat_items}
        if len(tags) < cfg.MIN_TOPIC_TAGS_PER_CATEGORY:
            violations.append(f"{cat}: only {len(tags)} topic tags (need >={cfg.MIN_TOPIC_TAGS_PER_CATEGORY})")
    total = len(items)
    if total > 0:
        trick_count = sum(1 for it in items if it.subtype in ("trick", "contradiction"))
        if trick_count / total > cfg.MAX_TRICK_RATIO:
            violations.append(f"trick/contradiction ratio {trick_count}/{total} = {trick_count/total:.2%} > {cfg.MAX_TRICK_RATIO:.0%}")
    for a, b, s, vtype in novelty_violations(items):
        violations.append(f"novelty ({vtype}): {a} <-> {b} cosine={s:.3f}")
    return violations


def marginal_ece_spread(existing_runs: list[RefRun], new_runs: list[RefRun]) -> float:
    return compute_ece_spread(existing_runs + new_runs) - compute_ece_spread(existing_runs)
