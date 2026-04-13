"""T39 — Graceful Degradation.

Tasks of increasing complexity (e.g., N-digit multiplication, sorting N
numbers).  We measure whether the model's confidence drops in step with its
accuracy as difficulty rises.  The headline metric *mean_inflection_lag*
captures the gap (in complexity steps) between the point where accuracy
collapses and the point where confidence collapses — lower is better.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_abstention_auroc, compute_auroc, pearson_r, spearman_rho
from ..llm_client import ConversationSession
import json
import numpy as np
import random


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_multiplication_item(n_digits: int, instance: int) -> dict:
    """Create an N-digit x N-digit multiplication problem."""
    rng = random.Random(42 + n_digits * 100 + instance)
    lo = 10 ** (n_digits - 1)
    hi = 10 ** n_digits - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    return {
        "task_type": "multiplication",
        "complexity": n_digits,
        "instance": instance,
        "problem": f"Compute {a} * {b}. Give only the numeric answer.",
        "answer": str(a * b),
    }


def _make_sorting_item(n_numbers: int, instance: int) -> dict:
    """Create a 'sort N random numbers' problem."""
    rng = random.Random(7 + n_numbers * 100 + instance)
    nums = [rng.randint(1, 9999) for _ in range(n_numbers)]
    sorted_nums = sorted(nums)
    return {
        "task_type": "sorting",
        "complexity": n_numbers,
        "instance": instance,
        "problem": (
            f"Sort the following {n_numbers} numbers in ascending order. "
            f"Return ONLY a comma-separated list of numbers.\n"
            f"Numbers: {', '.join(str(x) for x in nums)}"
        ),
        "answer": ", ".join(str(x) for x in sorted_nums),
    }


def _make_logic_chain_item(n_steps: int, instance: int) -> dict:
    """Create an N-step logical deduction problem."""
    rng = random.Random(13 + n_steps * 100 + instance)
    # Generate a chain of N logical deductions
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
             "Hank", "Iris", "Jack", "Kate", "Leo", "Mia", "Nick",
             "Olive", "Pat", "Quinn", "Rose", "Sam", "Tina",
             "Uma", "Vince", "Wendy", "Xander", "Yara", "Zack",
             "Amy", "Brian", "Clara", "Derek"]
    rng.shuffle(names)
    chosen = names[:min(n_steps + 1, len(names))]

    relations = ["taller than", "older than", "faster than", "richer than"]
    relation = rng.choice(relations)

    premises = []
    for i in range(n_steps):
        premises.append(f"{chosen[i]} is {relation} {chosen[i + 1]}")

    # The answer is the transitive conclusion
    question_text = (
        f"Given the following {n_steps} facts:\n"
        + "\n".join(f"{i+1}. {p}." for i, p in enumerate(premises))
        + f"\n\nWho is the {relation.split()[0]}est? Give ONLY the name."
    )

    return {
        "task_type": "logic_chain",
        "complexity": n_steps,
        "instance": instance,
        "problem": question_text,
        "answer": chosen[0],
    }


# Complexity levels per task type
_MULT_COMPLEXITIES = [1, 2, 3, 4, 5, 6, 7, 8]   # digits (widened)
_SORT_COMPLEXITIES = [3, 5, 8, 12, 20, 30, 50, 75]  # list length (widened)
_LOGIC_COMPLEXITIES = [2, 3, 5, 7, 10, 14, 18, 25]  # deduction steps (widened)

INSTANCES_PER_LEVEL = 3


class GracefulDegradation(BaseBenchmark):
    """T39: Graceful Degradation.

    Presents tasks at increasing difficulty and checks that the model's
    self-reported confidence tracks its actual accuracy decline.
    """

    benchmark_id = "t39"
    name = "Graceful Degradation"
    pillar = "Metacognitive Control"
    primary_metric = "mean_inflection_lag"

    # ------------------------------------------------------------------
    # Dataset — 3 task types x 8 levels x 3 instances = 72 items
    # ------------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        dataset = []
        for level_idx, n_digits in enumerate(_MULT_COMPLEXITIES):
            for inst in range(INSTANCES_PER_LEVEL):
                item = _make_multiplication_item(n_digits, inst)
                item["level_index"] = level_idx
                dataset.append(item)
        for level_idx, n_numbers in enumerate(_SORT_COMPLEXITIES):
            for inst in range(INSTANCES_PER_LEVEL):
                item = _make_sorting_item(n_numbers, inst)
                item["level_index"] = level_idx
                dataset.append(item)
        for level_idx, n_steps in enumerate(_LOGIC_COMPLEXITIES):
            for inst in range(INSTANCES_PER_LEVEL):
                item = _make_logic_chain_item(n_steps, inst)
                item["level_index"] = level_idx
                dataset.append(item)
        return dataset

    # ------------------------------------------------------------------
    # Per‑item execution
    # ------------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        problem = item["problem"]
        correct_answer = item["answer"]

        prompt = (
            f"{problem}\n\n"
            f"After providing your answer, also rate your confidence that "
            f"your answer is correct on a scale from 0 to 100.\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"answer": "<your answer>", "confidence": <0-100>}}'
        )
        resp = self.llm.prompt_json(prompt, temperature=0.0)
        model_answer = str(resp.get("answer", ""))
        confidence = float(resp.get("confidence", 50)) / 100.0
        confidence = max(0.0, min(1.0, confidence))

        correct = check_answer(model_answer, correct_answer, llm=self.llm, question=problem)

        return {
            "task_type": item["task_type"],
            "complexity": item["complexity"],
            "level_index": item["level_index"],
            "instance": item["instance"],
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "correct": correct,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {self.primary_metric: 1.0}

        inflection_lags = []
        per_type_metrics: dict[str, dict] = {}

        for task_type in ["multiplication", "sorting", "logic_chain"]:
            type_results = [r for r in results if r["task_type"] == task_type]
            if not type_results:
                continue

            # Group by complexity level
            levels = sorted(set(r["level_index"] for r in type_results))
            level_acc: list[float] = []
            level_conf: list[float] = []
            for lv in levels:
                lv_items = [r for r in type_results if r["level_index"] == lv]
                acc = float(np.mean([r["correct"] for r in lv_items]))
                conf = float(np.mean([r["confidence"] for r in lv_items]))
                level_acc.append(acc)
                level_conf.append(conf)

            # Find inflection points — the first level where metric drops
            # below 0.5 (or stays above for all levels).
            def _inflection(series: list[float], threshold: float = 0.5) -> int:
                for i, v in enumerate(series):
                    if v < threshold:
                        return i
                return len(series)  # never collapsed

            acc_inflection = _inflection(level_acc)
            conf_inflection = _inflection(level_conf)

            lag = abs(acc_inflection - conf_inflection)
            inflection_lags.append(lag)

            # Rank correlation between accuracy and confidence across levels
            rank_corr = spearman_rho(level_acc, level_conf) if len(levels) >= 3 else 0.0

            per_type_metrics[task_type] = {
                "level_accuracy": level_acc,
                "level_confidence": level_conf,
                "acc_inflection_level": acc_inflection,
                "conf_inflection_level": conf_inflection,
                "inflection_lag": lag,
                "rank_correlation": rank_corr,
            }

        mean_inflection_lag = float(np.mean(inflection_lags)) if inflection_lags else 1.0

        # Overall calibration metrics
        all_correct = [float(r["correct"]) for r in results]
        all_conf = [r["confidence"] for r in results]
        ece = compute_ece(all_correct, all_conf)
        rank_corr_overall = spearman_rho(all_correct, all_conf) if len(results) >= 3 else 0.0

        return {
            "mean_inflection_lag": mean_inflection_lag,
            "ece": ece,
            "rank_correlation": rank_corr_overall,
            "per_type": per_type_metrics,
            "n_items": len(results),
        }
