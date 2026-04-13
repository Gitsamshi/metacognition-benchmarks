"""T35 — Hedging Appropriateness: Model answers naturally (no structured
output forced for the answer itself).  Hedge language is analysed via
regex.  Good metacognition: hedge more on uncertain topics, less on
well-established facts."""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary, semantic_judge, llm_judge_uncertainty
from ..metrics import compute_ece, pearson_r, spearman_rho, compute_auroc
from ..llm_client import ConversationSession
import json
import re
import numpy as np


# ---------------------------------------------------------------------------
# Hedge detection
# ---------------------------------------------------------------------------

HEDGE_PATTERNS = [
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bmight\b",
    r"\bcould be\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bunlikely\b",
    r"\bi think\b",
    r"\bi believe\b",
    r"\bi\'m not sure\b",
    r"\bi am not sure\b",
    r"\bi\'m not certain\b",
    r"\bi am not certain\b",
    r"\bit seems\b",
    r"\bit appears\b",
    r"\bappears to be\b",
    r"\bit is possible\b",
    r"\bthere is a chance\b",
    r"\bas far as i know\b",
    r"\bto my knowledge\b",
    r"\bif i recall\b",
    r"\bif i remember\b",
    r"\bnot entirely sure\b",
    r"\bnot entirely certain\b",
    r"\bto some extent\b",
    r"\bin some cases\b",
    r"\bsomewhat\b",
    r"\broughly\b",
    r"\bapproximately\b",
    r"\baround\b",
    r"\babout\b",
    r"\btend to\b",
    r"\bsuggest(?:s|ed)?\b",
    r"\bunclear\b",
    r"\bdebated\b",
    r"\bcontroversial\b",
    r"\bnot well understood\b",
    r"\bmay or may not\b",
    r"\bit depends\b",
    r"\bgenerally\b",
    r"\btypically\b",
]

_COMPILED_HEDGES = [re.compile(p, re.IGNORECASE) for p in HEDGE_PATTERNS]


def count_hedges(text: str) -> int:
    """Count the number of hedge phrases in text."""
    total = 0
    for pattern in _COMPILED_HEDGES:
        total += len(pattern.findall(text))
    return total


def hedge_density(text: str) -> float:
    """Hedge count normalised by word count."""
    words = text.split()
    if not words:
        return 0.0
    return count_hedges(text) / len(words)


# ---------------------------------------------------------------------------
# Dataset — high-certainty (model should NOT hedge) and low-certainty
# (model SHOULD hedge)
# ---------------------------------------------------------------------------

_HIGH_CERTAINTY_QUESTIONS = [
    # Well-established scientific facts
    {"question": "What is the chemical formula for water?", "domain": "chemistry"},
    {"question": "How many continents are there on Earth?", "domain": "geography"},
    {"question": "What is the boiling point of water at sea level in Celsius?", "domain": "physics"},
    {"question": "What organ pumps blood through the human body?", "domain": "biology"},
    {"question": "What planet is closest to the Sun?", "domain": "astronomy"},
    {"question": "What is the square root of 144?", "domain": "mathematics"},
    {"question": "What gas do plants absorb during photosynthesis?", "domain": "biology"},
    {"question": "Who wrote Romeo and Juliet?", "domain": "literature"},
    {"question": "What is the speed of light in a vacuum, approximately?", "domain": "physics"},
    {"question": "In which year did World War II end?", "domain": "history"},
    {"question": "What is the largest planet in our solar system?", "domain": "astronomy"},
    {"question": "How many bones are in the adult human body?", "domain": "biology"},
    {"question": "What element has the atomic number 1?", "domain": "chemistry"},
    {"question": "What is the capital of France?", "domain": "geography"},
    {"question": "What force keeps planets in orbit around the Sun?", "domain": "physics"},
    {"question": "What is the value of pi to two decimal places?", "domain": "mathematics"},
    {"question": "What is the primary function of red blood cells?", "domain": "biology"},
    {"question": "Who painted the Mona Lisa?", "domain": "art"},
    {"question": "What is the chemical symbol for gold?", "domain": "chemistry"},
    {"question": "How many days are in a leap year?", "domain": "general knowledge"},
    {"question": "What is the tallest mountain on Earth?", "domain": "geography"},
    {"question": "What does DNA stand for?", "domain": "biology"},
    {"question": "Who developed the theory of general relativity?", "domain": "physics"},
    {"question": "What is the freezing point of water in Fahrenheit?", "domain": "physics"},
    {"question": "In what country are the Pyramids of Giza located?", "domain": "geography"},
    {"question": "What is 7 multiplied by 8?", "domain": "mathematics"},
    {"question": "What language has the most native speakers worldwide?", "domain": "linguistics"},
    {"question": "What is the hardest natural substance on Earth?", "domain": "geology"},
    {"question": "What is the main component of the Sun?", "domain": "astronomy"},
    {"question": "What year was the Declaration of Independence signed?", "domain": "history"},
]

_LOW_CERTAINTY_QUESTIONS = [
    # Open questions, cutting-edge research, speculative topics
    {"question": "What is the most likely resolution to the Fermi paradox?", "domain": "astrobiology"},
    {"question": "Will quantum computers ever fully replace classical computers for everyday tasks?", "domain": "computing"},
    {"question": "What is the leading explanation for the accelerating expansion of the universe?", "domain": "cosmology"},
    {"question": "What role does the gut microbiome play in mental health disorders?", "domain": "medicine"},
    {"question": "Is P equal to NP?", "domain": "computer science"},
    {"question": "What are the long-term societal effects of widespread AI adoption?",  "domain": "technology"},
    {"question": "What caused the Late Bronze Age collapse?", "domain": "history"},
    {"question": "How did consciousness evolve and what is its adaptive function?", "domain": "neuroscience"},
    {"question": "Will fusion power become commercially viable within the next 30 years?", "domain": "energy"},
    {"question": "What is the nature of dark matter?", "domain": "physics"},
    {"question": "Could life exist in the subsurface oceans of Europa?", "domain": "astrobiology"},
    {"question": "What are the primary drivers of the current mass extinction event?", "domain": "ecology"},
    {"question": "How will climate change affect global food production by 2100?", "domain": "environmental science"},
    {"question": "What is the best economic system for reducing inequality while promoting growth?", "domain": "economics"},
    {"question": "Why do some patients respond to placebos while others do not?", "domain": "medicine"},
    {"question": "Is the many-worlds interpretation of quantum mechanics correct?", "domain": "physics"},
    {"question": "What explains the gap between rich and poor nations?", "domain": "economics"},
    {"question": "How close are we to achieving artificial general intelligence?", "domain": "AI"},
    {"question": "What is the optimal strategy for governing global commons like the oceans?", "domain": "political science"},
    {"question": "Will CRISPR gene editing eliminate genetic diseases within a generation?", "domain": "genetics"},
    {"question": "What is the biological basis of free will, if any?", "domain": "neuroscience"},
    {"question": "How will the rise of remote work reshape cities over the next 50 years?", "domain": "urban planning"},
    {"question": "What is the mechanism behind ball lightning?", "domain": "physics"},
    {"question": "Can we ever fully decode the Voynich Manuscript?", "domain": "linguistics"},
    {"question": "What is the optimal diet for longevity?", "domain": "nutrition"},
    {"question": "Will blockchain technology fundamentally transform governance?", "domain": "technology"},
    {"question": "What caused the Cambrian Explosion?", "domain": "paleontology"},
    {"question": "Could we ever reverse human ageing?", "domain": "biology"},
    {"question": "What are the ethical limits of human genetic enhancement?", "domain": "bioethics"},
    {"question": "What were the true causes of the fall of the Roman Empire?", "domain": "history"},
]


class HedgingAppropriateness(BaseBenchmark):
    benchmark_id = "t35"
    name = "Hedging Appropriateness"
    pillar = "Knowledge Boundary Awareness"
    primary_metric = "hedge_appropriateness"

    # ---- dataset ----
    def generate_dataset(self) -> list[dict]:
        items = []
        for q in _HIGH_CERTAINTY_QUESTIONS:
            items.append({
                "question": q["question"],
                "domain": q["domain"],
                "certainty_group": "high_certainty",
            })
        for q in _LOW_CERTAINTY_QUESTIONS:
            items.append({
                "question": q["question"],
                "domain": q["domain"],
                "certainty_group": "low_certainty",
            })
        return items  # 30 + 30 = 60

    # ---- per-item ----
    def run_item(self, item: dict) -> dict:
        # Deliberately ask for a natural-language answer, no JSON, no
        # confidence scores — we want to measure organic hedging behaviour.
        prompt = (
            f'{item["question"]}\n\n'
            "Please provide a clear, informative answer."
        )
        response = self.llm.prompt(prompt)

        n_hedges = count_hedges(response)
        density = hedge_density(response)
        word_count = len(response.split())

        # LLM-based semantic hedge detection (primary metric)
        llm_uncertainty_score = llm_judge_uncertainty(self.llm, response)

        return {
            "question": item["question"],
            "domain": item["domain"],
            "certainty_group": item["certainty_group"],
            "response": response,
            "hedge_count": n_hedges,
            "hedge_density": density,
            "word_count": word_count,
            "llm_uncertainty_score": llm_uncertainty_score,
        }

    # ---- aggregate ----
    def aggregate(self, results: list[dict]) -> dict:
        high = [r for r in results if r["certainty_group"] == "high_certainty"]
        low = [r for r in results if r["certainty_group"] == "low_certainty"]

        # Regex-based metrics (secondary)
        high_density = float(np.mean([r["hedge_density"] for r in high])) if high else 0.0
        low_density = float(np.mean([r["hedge_density"] for r in low])) if low else 0.0
        regex_hedge_appropriateness = low_density - high_density

        # LLM-based semantic hedge metrics (primary)
        high_llm_score = float(np.mean([r["llm_uncertainty_score"] for r in high])) if high else 0.0
        low_llm_score = float(np.mean([r["llm_uncertainty_score"] for r in low])) if low else 0.0
        llm_hedge_appropriateness = (low_llm_score - high_llm_score) / 5.0  # normalise to ~[0,1]

        # Use LLM-based as primary
        hedge_appropriateness = llm_hedge_appropriateness

        return {
            "hedge_appropriateness": float(hedge_appropriateness),
            "llm_hedge_appropriateness": float(llm_hedge_appropriateness),
            "regex_hedge_appropriateness": float(regex_hedge_appropriateness),
            "high_certainty_llm_uncertainty": high_llm_score,
            "low_certainty_llm_uncertainty": low_llm_score,
            "high_certainty_hedge_density": high_density,
            "low_certainty_hedge_density": low_density,
            "high_certainty_mean_hedges": float(np.mean([r["hedge_count"] for r in high])) if high else 0.0,
            "low_certainty_mean_hedges": float(np.mean([r["hedge_count"] for r in low])) if low else 0.0,
            "high_certainty_mean_words": float(np.mean([r["word_count"] for r in high])) if high else 0.0,
            "low_certainty_mean_words": float(np.mean([r["word_count"] for r in low])) if low else 0.0,
            "n_high_certainty": len(high),
            "n_low_certainty": len(low),
        }
