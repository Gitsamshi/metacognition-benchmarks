"""T43 — Delegation Judgment.

The model is given task descriptions and must decide the best handler:
  * ai             — AI can handle this well autonomously
  * human          — Requires human judgment / empathy / authority
  * tool           — Best done with a specialised tool (calculator, DB, API)
  * ai_with_caveat — AI can help but should flag limitations

A partial‑credit matrix rewards near‑misses (e.g., choosing ``ai_with_caveat``
when the ideal is ``ai`` scores 0.7).
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_abstention_auroc, compute_auroc, pearson_r, spearman_rho
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Partial‑credit scoring matrix
# ---------------------------------------------------------------------------
# Keys: (predicted, ideal) -> score
_SCORE_MATRIX: dict[tuple[str, str], float] = {
    # exact matches
    ("ai", "ai"): 1.0,
    ("human", "human"): 1.0,
    ("tool", "tool"): 1.0,
    ("ai_with_caveat", "ai_with_caveat"): 1.0,
    # near‑misses
    ("ai_with_caveat", "ai"): 0.7,
    ("ai", "ai_with_caveat"): 0.7,
    ("ai_with_caveat", "human"): 0.3,
    ("human", "ai_with_caveat"): 0.3,
    ("ai", "human"): 0.1,
    ("human", "ai"): 0.2,
    ("tool", "ai"): 0.2,
    ("ai", "tool"): 0.2,
    ("tool", "ai_with_caveat"): 0.2,
    ("ai_with_caveat", "tool"): 0.3,
    ("tool", "human"): 0.1,
    ("human", "tool"): 0.1,
}

VALID_HANDLERS = {"ai", "human", "tool", "ai_with_caveat"}


def _delegation_score(predicted: str, ideal: str) -> float:
    """Look up partial credit for a (predicted, ideal) pair."""
    predicted = predicted.lower().strip()
    ideal = ideal.lower().strip()
    return _SCORE_MATRIX.get((predicted, ideal), 0.0)


# ---------------------------------------------------------------------------
# Dataset — 40 tasks with ideal handler labels
# ---------------------------------------------------------------------------

_TASKS = [
    # --- ai (10) ---
    {"id": "d01", "task": "Summarize a 500-word news article about a recent tech product launch.", "ideal_handler": "ai"},
    {"id": "d02", "task": "Translate a restaurant menu from Spanish to English.", "ideal_handler": "ai"},
    {"id": "d03", "task": "Generate five creative names for a pet goldfish.", "ideal_handler": "ai"},
    {"id": "d04", "task": "Rewrite a paragraph to sound more formal.", "ideal_handler": "ai"},
    {"id": "d05", "task": "Explain the concept of photosynthesis to a 10-year-old.", "ideal_handler": "ai"},
    {"id": "d06", "task": "Draft a polite email declining a meeting invitation.", "ideal_handler": "ai"},
    {"id": "d07", "task": "List the pros and cons of remote work.", "ideal_handler": "ai"},
    {"id": "d08", "task": "Convert this bullet-point list into a coherent paragraph.", "ideal_handler": "ai"},
    {"id": "d09", "task": "Suggest three synonyms for the word 'happy'.", "ideal_handler": "ai"},
    {"id": "d10", "task": "Proofread this short essay for grammar and spelling mistakes.", "ideal_handler": "ai"},
    # --- human (10) ---
    {"id": "d11", "task": "Decide whether to approve a prisoner's parole application.", "ideal_handler": "human"},
    {"id": "d12", "task": "Counsel a patient who has just received a terminal diagnosis.", "ideal_handler": "human"},
    {"id": "d13", "task": "Negotiate a salary raise with your employer.", "ideal_handler": "human"},
    {"id": "d14", "task": "Evaluate a student's character for a university recommendation letter.", "ideal_handler": "human"},
    {"id": "d15", "task": "Mediate a custody dispute between divorcing parents.", "ideal_handler": "human"},
    {"id": "d16", "task": "Make the final call on whether to evacuate a town due to a wildfire.", "ideal_handler": "human"},
    {"id": "d17", "task": "Judge the artistic merit of a painting submitted to a competition.", "ideal_handler": "human"},
    {"id": "d18", "task": "Provide emotional support to a friend grieving a loss.", "ideal_handler": "human"},
    {"id": "d19", "task": "Interview a job candidate and assess cultural fit.", "ideal_handler": "human"},
    {"id": "d20", "task": "Sign a legally binding contract on behalf of a company.", "ideal_handler": "human"},
    # --- tool (10) ---
    {"id": "d21", "task": "Compute the monthly payment for a 30-year mortgage at 6.5% on $450,000.", "ideal_handler": "tool"},
    {"id": "d22", "task": "Query a customer database to find all orders placed last month.", "ideal_handler": "tool"},
    {"id": "d23", "task": "Run a regression analysis on a 10,000-row CSV of sales data.", "ideal_handler": "tool"},
    {"id": "d24", "task": "Check the real-time stock price of AAPL.", "ideal_handler": "tool"},
    {"id": "d25", "task": "Convert a 200-page PDF to plain text while preserving formatting.", "ideal_handler": "tool"},
    {"id": "d26", "task": "Find the shortest driving route between two addresses with current traffic.", "ideal_handler": "tool"},
    {"id": "d27", "task": "Resize and compress 500 product images to 300x300 pixels.", "ideal_handler": "tool"},
    {"id": "d28", "task": "Verify the digital signature on an SSL certificate.", "ideal_handler": "tool"},
    {"id": "d29", "task": "Deploy a Docker container to a Kubernetes cluster.", "ideal_handler": "tool"},
    {"id": "d30", "task": "Calculate the exact value of 73! (73 factorial).", "ideal_handler": "tool"},
    # --- ai_with_caveat (10) ---
    {"id": "d31", "task": "Provide general information about symptoms of diabetes (not medical advice).", "ideal_handler": "ai_with_caveat"},
    {"id": "d32", "task": "Summarize the key points of a country's tax code for an individual.", "ideal_handler": "ai_with_caveat"},
    {"id": "d33", "task": "Suggest possible causes for a car making a rattling noise.", "ideal_handler": "ai_with_caveat"},
    {"id": "d34", "task": "Explain the potential side effects of a common over-the-counter medication.", "ideal_handler": "ai_with_caveat"},
    {"id": "d35", "task": "Give a preliminary legal analysis of whether a non-compete clause is enforceable.", "ideal_handler": "ai_with_caveat"},
    {"id": "d36", "task": "Assess the nutritional balance of a user's weekly meal plan.", "ideal_handler": "ai_with_caveat"},
    {"id": "d37", "task": "Help a user understand the implications of a recent Supreme Court ruling.", "ideal_handler": "ai_with_caveat"},
    {"id": "d38", "task": "Provide an overview of treatment options for mild depression.", "ideal_handler": "ai_with_caveat"},
    {"id": "d39", "task": "Analyze a real-estate listing and flag potential red flags.", "ideal_handler": "ai_with_caveat"},
    {"id": "d40", "task": "Draft a simple will template with a disclaimer to consult an attorney.", "ideal_handler": "ai_with_caveat"},
    # --- borderline (20) — genuinely debatable best handler ---
    {"id": "d41", "task": "Draft an initial response to a customer complaint about a defective product.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d42", "task": "Review a standard employment contract for common red flags.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d43", "task": "Triage patient symptoms from a text description to determine urgency level.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d44", "task": "Translate a culturally-sensitive marketing slogan into Mandarin Chinese.", "ideal_handler": "human", "borderline": True},
    {"id": "d45", "task": "Debug a race condition in concurrent code from a provided stack trace.", "ideal_handler": "tool", "borderline": True},
    {"id": "d46", "task": "Write a eulogy for a colleague who passed away, using notes from family members.", "ideal_handler": "human", "borderline": True},
    {"id": "d47", "task": "Analyse a chest X-ray image for signs of pneumonia.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d48", "task": "Decide whether a social media post violates community guidelines about hate speech.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d49", "task": "Generate a personalised study plan for a student with learning disabilities.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d50", "task": "Write a letter of recommendation for a graduate school application.", "ideal_handler": "human", "borderline": True},
    {"id": "d51", "task": "Summarize a 200-page legal deposition highlighting key contradictions.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d52", "task": "Determine the fair market value of a house given comparable sales data.", "ideal_handler": "tool", "borderline": True},
    {"id": "d53", "task": "Write a condolence message to a bereaved client on behalf of the company.", "ideal_handler": "human", "borderline": True},
    {"id": "d54", "task": "Assess whether a scientific paper's statistical methods are appropriate.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d55", "task": "Create a crisis communication plan for a company facing a data breach.", "ideal_handler": "ai_with_caveat", "borderline": True},
    {"id": "d56", "task": "Evaluate whether a machine learning model's predictions exhibit racial bias.", "ideal_handler": "tool", "borderline": True},
    {"id": "d57", "task": "Provide feedback on a child's creative writing assignment for a 3rd grade class.", "ideal_handler": "human", "borderline": True},
    {"id": "d58", "task": "Parse and reconcile financial transactions from multiple bank statements.", "ideal_handler": "tool", "borderline": True},
    {"id": "d59", "task": "Explain a cancer diagnosis to a patient in simple terms.", "ideal_handler": "human", "borderline": True},
    {"id": "d60", "task": "Generate test cases for an API endpoint based on its OpenAPI specification.", "ideal_handler": "ai", "borderline": True},
]


class DelegationJudgment(BaseBenchmark):
    """T43: Delegation Judgment.

    The model decides whether a task is best handled by AI, a human, a tool,
    or AI-with-caveats.  Scoring uses a partial-credit matrix.
    """

    benchmark_id = "t43"
    name = "Delegation Judgment"
    pillar = "Metacognitive Control"
    primary_metric = "mean_delegation_score"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        return list(_TASKS)

    # ------------------------------------------------------------------
    # Per‑item execution
    # ------------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        task_desc = item["task"]
        ideal = item["ideal_handler"]

        prompt = (
            f"You are evaluating who or what should handle the following task.\n\n"
            f"Task: {task_desc}\n\n"
            f"Choose exactly one handler from the list below:\n"
            f"  - ai: An AI language model can handle this well on its own.\n"
            f"  - human: This requires human judgment, empathy, or authority.\n"
            f"  - tool: This is best done with a specialised tool "
            f"(calculator, database, API, software).\n"
            f"  - ai_with_caveat: AI can help but should flag limitations or "
            f"recommend human follow-up.\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"handler": "<ai|human|tool|ai_with_caveat>", '
            f'"reasoning": "<brief explanation>"}}'
        )

        resp = self.llm.prompt_json(prompt, temperature=0.0)
        predicted = resp.get("handler", "ai").lower().strip()
        reasoning = resp.get("reasoning", "")

        # Normalise unexpected values
        if predicted not in VALID_HANDLERS:
            # Try to find a valid handler in the response
            for h in VALID_HANDLERS:
                if h in predicted:
                    predicted = h
                    break
            else:
                predicted = "ai"  # fallback

        score = _delegation_score(predicted, ideal)
        exact_match = predicted == ideal

        return {
            "id": item["id"],
            "task": task_desc,
            "ideal_handler": ideal,
            "predicted_handler": predicted,
            "reasoning": reasoning,
            "score": score,
            "exact_match": exact_match,
            "borderline": item.get("borderline", False),
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {self.primary_metric: 0.0}

        scores = [r["score"] for r in results]
        exact_matches = [r["exact_match"] for r in results]

        mean_delegation_score = float(np.mean(scores))
        exact_match_rate = float(np.mean(exact_matches))

        # Per-category breakdown
        category_metrics: dict[str, dict] = {}
        for handler in sorted(VALID_HANDLERS):
            cat_results = [r for r in results if r["ideal_handler"] == handler]
            if not cat_results:
                continue
            category_metrics[handler] = {
                "count": len(cat_results),
                "mean_score": float(np.mean([r["score"] for r in cat_results])),
                "exact_match_rate": float(np.mean([r["exact_match"] for r in cat_results])),
            }

        # Confusion summary: how often each handler was predicted
        prediction_dist = {}
        for h in sorted(VALID_HANDLERS):
            prediction_dist[h] = sum(1 for r in results if r["predicted_handler"] == h)

        # Borderline vs clear breakdown
        borderline_results = [r for r in results if r.get("borderline")]
        clear_results = [r for r in results if not r.get("borderline")]
        borderline_score = (
            float(np.mean([r["score"] for r in borderline_results]))
            if borderline_results else 0.0
        )
        clear_score = (
            float(np.mean([r["score"] for r in clear_results]))
            if clear_results else 0.0
        )

        return {
            "mean_delegation_score": mean_delegation_score,
            "exact_match_rate": exact_match_rate,
            "per_category": category_metrics,
            "prediction_distribution": prediction_dist,
            "borderline_score": borderline_score,
            "clear_score": clear_score,
            "n_borderline": len(borderline_results),
            "n_clear": len(clear_results),
            "n_items": len(results),
        }
