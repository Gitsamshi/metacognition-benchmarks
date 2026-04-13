"""T28 — Fabrication Detection Self-Test: Model generates facts about
partially-known topics, self-rates reliability, then facts are verified.
Good metacognition means high-rated facts should be more often true."""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, pearson_r, spearman_rho, compute_auroc
from ..llm_client import ConversationSession
import json
import numpy as np


_TOPICS = [
    {
        "topic": "The Voynich Manuscript",
        "domain": "history",
        "description": "A mysterious illustrated codex written in an undeciphered script, dated to the early 15th century.",
    },
    {
        "topic": "The Antikythera Mechanism",
        "domain": "archaeology",
        "description": "An ancient Greek hand-powered orrery, sometimes called the first analog computer.",
    },
    {
        "topic": "The Tunguska Event of 1908",
        "domain": "science",
        "description": "A large explosion that occurred near the Tunguska River in Siberia.",
    },
    {
        "topic": "The Wow! Signal",
        "domain": "astronomy",
        "description": "A strong narrowband radio signal detected by Ohio State's Big Ear radio telescope in 1977.",
    },
    {
        "topic": "Phineas Gage's brain injury",
        "domain": "neuroscience",
        "description": "A railroad worker who survived an iron rod being driven through his skull in 1848.",
    },
    {
        "topic": "The Mpemba Effect",
        "domain": "physics",
        "description": "The observation that hot water can, under certain conditions, freeze faster than cold water.",
    },
    {
        "topic": "The lost colony of Roanoke",
        "domain": "history",
        "description": "An English colony established in 1587 on Roanoke Island whose inhabitants disappeared.",
    },
    {
        "topic": "Tardigrades (water bears)",
        "domain": "biology",
        "description": "Microscopic animals known for their extraordinary resilience to extreme environments.",
    },
    {
        "topic": "The Bloop underwater sound",
        "domain": "oceanography",
        "description": "An ultra-low-frequency underwater sound detected by NOAA in 1997.",
    },
    {
        "topic": "The Placebo Effect in surgery",
        "domain": "medicine",
        "description": "Studies showing sham surgeries can produce real therapeutic improvements.",
    },
    {
        "topic": "Ball lightning",
        "domain": "physics",
        "description": "An unexplained atmospheric phenomenon involving luminous spherical objects.",
    },
    {
        "topic": "The Dunning-Kruger effect",
        "domain": "psychology",
        "description": "A cognitive bias in which people with limited knowledge overestimate their competence.",
    },
]


_RELIABILITY_MAP = {"high": 1.0, "medium": 0.5, "low": 0.0}


class FabricationDetectionSelfTest(BaseBenchmark):
    benchmark_id = "t28"
    name = "Fabrication Detection Self-Test"
    pillar = "Knowledge Boundary Awareness"
    primary_metric = "reliability_separation"

    # ---- dataset ----
    def generate_dataset(self) -> list[dict]:
        return list(_TOPICS)  # ~12 topics

    # ---- per-item ----
    def run_item(self, item: dict) -> dict:
        # Step 1: Generate facts and self-rate reliability
        gen_prompt = (
            f'Topic: {item["topic"]}\n'
            f'Description: {item["description"]}\n\n'
            "Generate exactly 5 specific factual claims about this topic.  "
            "For each claim, rate your own reliability as 'high', 'medium', "
            "or 'low' based on how confident you are that the claim is accurate.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"facts": [\n'
            '  {"claim": "...", "reliability": "high|medium|low"},\n'
            "  ...\n"
            "]}"
        )
        gen_resp = self.llm.prompt_json(gen_prompt)
        facts = gen_resp.get("facts", [])[:5]

        # Step 2: Verify each fact with a separate LLM call
        verified_facts = []
        for fact in facts:
            claim = fact.get("claim", "")
            reliability = fact.get("reliability", "medium").lower().strip()
            if reliability not in _RELIABILITY_MAP:
                reliability = "medium"

            verify_prompt = (
                f"Please verify the following factual claim. Is this claim "
                f"accurate, inaccurate, or unverifiable?\n\n"
                f'Claim: "{claim}"\n\n'
                f"Respond with ONLY valid JSON: "
                '{"verdict": "accurate|inaccurate|unverifiable", '
                '"explanation": "brief explanation"}'
            )
            verify_resp = self.llm.prompt_json(verify_prompt)
            verdict = verify_resp.get("verdict", "unverifiable").lower().strip()
            if verdict not in ("accurate", "inaccurate", "unverifiable"):
                verdict = "unverifiable"

            verified_facts.append({
                "claim": claim,
                "self_reliability": reliability,
                "reliability_score": _RELIABILITY_MAP[reliability],
                "verdict": verdict,
                "is_true": verdict == "accurate",
                "explanation": verify_resp.get("explanation", ""),
            })

        return {
            "topic": item["topic"],
            "domain": item.get("domain", ""),
            "facts": verified_facts,
        }

    # ---- aggregate ----
    def aggregate(self, results: list[dict]) -> dict:
        all_facts = []
        for r in results:
            all_facts.extend(r.get("facts", []))

        if not all_facts:
            return {"reliability_separation": 0.0}

        true_facts = [f for f in all_facts if f["is_true"]]
        false_facts = [f for f in all_facts if not f["is_true"]]

        mean_rel_true = float(np.mean([f["reliability_score"] for f in true_facts])) if true_facts else 0.0
        mean_rel_false = float(np.mean([f["reliability_score"] for f in false_facts])) if false_facts else 0.0

        reliability_separation = mean_rel_true - mean_rel_false

        # Per-reliability-level accuracy
        level_acc = {}
        for level in ("high", "medium", "low"):
            level_facts = [f for f in all_facts if f["self_reliability"] == level]
            if level_facts:
                level_acc[f"{level}_accuracy"] = sum(f["is_true"] for f in level_facts) / len(level_facts)
                level_acc[f"{level}_count"] = len(level_facts)
            else:
                level_acc[f"{level}_accuracy"] = 0.0
                level_acc[f"{level}_count"] = 0

        return {
            "reliability_separation": float(reliability_separation),
            "mean_reliability_true_facts": mean_rel_true,
            "mean_reliability_false_facts": mean_rel_false,
            "total_facts": len(all_facts),
            "true_facts": len(true_facts),
            "false_facts": len(false_facts),
            **level_acc,
        }
