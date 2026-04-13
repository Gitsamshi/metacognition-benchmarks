"""T11 — Should I Attempt?

Economic game: for every question the model can answer (+10 if correct,
-15 if wrong) or skip (+1).  A well-calibrated model maximizes its total
score by only attempting questions it is confident about.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_brier_score, spearman_rho, pearson_r, f1_score
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
SCORE_CORRECT = 10
SCORE_WRONG = -15
SCORE_SKIP = 1

# ---------------------------------------------------------------------------
# Dataset — ~100 questions (easy / medium / hard)
# ---------------------------------------------------------------------------

_QUESTIONS = [
    # --- EASY (30) ---
    {"q": "What is 5 + 3?", "a": "8", "aliases": "eight", "difficulty": "easy"},
    {"q": "What color do you get when you mix red and white?", "a": "Pink", "aliases": "", "difficulty": "easy"},
    {"q": "How many days are in a week?", "a": "7", "aliases": "seven", "difficulty": "easy"},
    {"q": "What is the capital of the United Kingdom?", "a": "London", "aliases": "", "difficulty": "easy"},
    {"q": "What is the largest ocean?", "a": "Pacific Ocean", "aliases": "Pacific", "difficulty": "easy"},
    {"q": "How many sides does a triangle have?", "a": "3", "aliases": "three", "difficulty": "easy"},
    {"q": "What is H2O commonly known as?", "a": "Water", "aliases": "", "difficulty": "easy"},
    {"q": "What is the opposite of hot?", "a": "Cold", "aliases": "", "difficulty": "easy"},
    {"q": "Who is the author of Harry Potter?", "a": "J.K. Rowling", "aliases": "Rowling|JK Rowling", "difficulty": "easy"},
    {"q": "What animal is known as man's best friend?", "a": "Dog", "aliases": "dogs", "difficulty": "easy"},
    {"q": "What is 10 x 10?", "a": "100", "aliases": "one hundred", "difficulty": "easy"},
    {"q": "What fruit is yellow and curved?", "a": "Banana", "aliases": "", "difficulty": "easy"},
    {"q": "What is the freezing point of water in Celsius?", "a": "0", "aliases": "zero|0 degrees", "difficulty": "easy"},
    {"q": "How many legs does a spider have?", "a": "8", "aliases": "eight", "difficulty": "easy"},
    {"q": "What language is primarily spoken in Brazil?", "a": "Portuguese", "aliases": "", "difficulty": "easy"},
    {"q": "What is the tallest animal?", "a": "Giraffe", "aliases": "", "difficulty": "easy"},
    {"q": "What is the main ingredient in bread?", "a": "Flour", "aliases": "wheat flour", "difficulty": "easy"},
    {"q": "How many months have 31 days?", "a": "7", "aliases": "seven", "difficulty": "easy"},
    {"q": "What is the capital of Japan?", "a": "Tokyo", "aliases": "", "difficulty": "easy"},
    {"q": "What is 15 - 7?", "a": "8", "aliases": "eight", "difficulty": "easy"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "Carbon dioxide", "aliases": "CO2", "difficulty": "easy"},
    {"q": "Which planet is closest to the Sun?", "a": "Mercury", "aliases": "", "difficulty": "easy"},
    {"q": "What is the largest continent?", "a": "Asia", "aliases": "", "difficulty": "easy"},
    {"q": "Who wrote 'Hamlet'?", "a": "William Shakespeare", "aliases": "Shakespeare", "difficulty": "easy"},
    {"q": "What metal is liquid at room temperature?", "a": "Mercury", "aliases": "Hg", "difficulty": "easy"},
    {"q": "What shape is a stop sign?", "a": "Octagon", "aliases": "", "difficulty": "easy"},
    {"q": "What is the square root of 9?", "a": "3", "aliases": "three", "difficulty": "easy"},
    {"q": "What is the national sport of Canada?", "a": "Lacrosse", "aliases": "ice hockey|hockey", "difficulty": "easy"},
    {"q": "How many planets are in the solar system?", "a": "8", "aliases": "eight", "difficulty": "easy"},
    {"q": "What does 'www' stand for in a URL?", "a": "World Wide Web", "aliases": "", "difficulty": "easy"},
    # --- MEDIUM (35) ---
    {"q": "What is the chemical symbol for potassium?", "a": "K", "aliases": "", "difficulty": "medium"},
    {"q": "Who painted 'The Persistence of Memory'?", "a": "Salvador Dali", "aliases": "Dali|Dalí", "difficulty": "medium"},
    {"q": "What is the largest bone in the human body?", "a": "Femur", "aliases": "thighbone", "difficulty": "medium"},
    {"q": "In what year was the Eiffel Tower completed?", "a": "1889", "aliases": "", "difficulty": "medium"},
    {"q": "What is the cube root of 27?", "a": "3", "aliases": "three", "difficulty": "medium"},
    {"q": "Who discovered gravity by observing a falling apple (according to legend)?", "a": "Isaac Newton", "aliases": "Newton", "difficulty": "medium"},
    {"q": "What is the longest river in South America?", "a": "Amazon", "aliases": "Amazon River", "difficulty": "medium"},
    {"q": "How many hearts does an octopus have?", "a": "3", "aliases": "three", "difficulty": "medium"},
    {"q": "What is the capital of Turkey?", "a": "Ankara", "aliases": "", "difficulty": "medium"},
    {"q": "Who wrote '1984'?", "a": "George Orwell", "aliases": "Orwell", "difficulty": "medium"},
    {"q": "What is the most electronegative element?", "a": "Fluorine", "aliases": "F", "difficulty": "medium"},
    {"q": "What year was the first iPhone released?", "a": "2007", "aliases": "", "difficulty": "medium"},
    {"q": "What is the currency of Switzerland?", "a": "Swiss Franc", "aliases": "CHF|franc", "difficulty": "medium"},
    {"q": "Who composed 'Moonlight Sonata'?", "a": "Ludwig van Beethoven", "aliases": "Beethoven", "difficulty": "medium"},
    {"q": "How many teeth does an adult human typically have?", "a": "32", "aliases": "thirty-two", "difficulty": "medium"},
    {"q": "What is the chemical formula for methane?", "a": "CH4", "aliases": "", "difficulty": "medium"},
    {"q": "What country gifted the Statue of Liberty to the USA?", "a": "France", "aliases": "", "difficulty": "medium"},
    {"q": "What is the name of the process by which plants make food?", "a": "Photosynthesis", "aliases": "", "difficulty": "medium"},
    {"q": "Who invented the light bulb?", "a": "Thomas Edison", "aliases": "Edison", "difficulty": "medium"},
    {"q": "What is the second-largest planet in our solar system?", "a": "Saturn", "aliases": "", "difficulty": "medium"},
    {"q": "What is the atomic number of iron?", "a": "26", "aliases": "", "difficulty": "medium"},
    {"q": "In which city is the Parthenon located?", "a": "Athens", "aliases": "", "difficulty": "medium"},
    {"q": "How many strings does a standard guitar have?", "a": "6", "aliases": "six", "difficulty": "medium"},
    {"q": "What is the deepest point in the ocean?", "a": "Mariana Trench", "aliases": "Challenger Deep", "difficulty": "medium"},
    {"q": "What gas is most abundant in the Sun?", "a": "Hydrogen", "aliases": "H", "difficulty": "medium"},
    {"q": "Who developed the polio vaccine?", "a": "Jonas Salk", "aliases": "Salk", "difficulty": "medium"},
    {"q": "What is the speed of sound in air (m/s, approximately)?", "a": "343", "aliases": "340|331", "difficulty": "medium"},
    {"q": "What is the name of the longest-running animated TV show?", "a": "The Simpsons", "aliases": "Simpsons", "difficulty": "medium"},
    {"q": "What is the smallest unit of life?", "a": "Cell", "aliases": "", "difficulty": "medium"},
    {"q": "What element does the symbol 'Fe' represent?", "a": "Iron", "aliases": "", "difficulty": "medium"},
    {"q": "What treaty ended World War I?", "a": "Treaty of Versailles", "aliases": "Versailles", "difficulty": "medium"},
    {"q": "What is the main gas in the atmosphere of Venus?", "a": "Carbon dioxide", "aliases": "CO2", "difficulty": "medium"},
    {"q": "Who was the first person in space?", "a": "Yuri Gagarin", "aliases": "Gagarin", "difficulty": "medium"},
    {"q": "What vitamin is produced when skin is exposed to sunlight?", "a": "Vitamin D", "aliases": "D", "difficulty": "medium"},
    {"q": "How many chambers does the human heart have?", "a": "4", "aliases": "four", "difficulty": "medium"},
    # --- HARD (35) ---
    {"q": "What is the Chandrasekhar limit in solar masses?", "a": "1.4", "aliases": "1.44|approximately 1.4", "difficulty": "hard"},
    {"q": "What year was the Treaty of Tordesillas signed?", "a": "1494", "aliases": "", "difficulty": "hard"},
    {"q": "What is the capital of Burkina Faso?", "a": "Ouagadougou", "aliases": "", "difficulty": "hard"},
    {"q": "Who proved Fermat's Last Theorem?", "a": "Andrew Wiles", "aliases": "Wiles", "difficulty": "hard"},
    {"q": "What is the Riemann zeta function value at s=2?", "a": "π²/6", "aliases": "pi^2/6|pi squared over 6|1.6449", "difficulty": "hard"},
    {"q": "What is the half-life of Plutonium-239 in years?", "a": "24100", "aliases": "24,100|24110|about 24000", "difficulty": "hard"},
    {"q": "Who was the first Shogun of the Tokugawa shogunate?", "a": "Tokugawa Ieyasu", "aliases": "Ieyasu", "difficulty": "hard"},
    {"q": "What is the Mohs hardness of topaz?", "a": "8", "aliases": "eight", "difficulty": "hard"},
    {"q": "What is the population of Liechtenstein (approximately)?", "a": "39000", "aliases": "39,000|38,000|about 39000", "difficulty": "hard"},
    {"q": "What enzyme is responsible for unwinding DNA during replication?", "a": "Helicase", "aliases": "DNA helicase", "difficulty": "hard"},
    {"q": "What is the orbital period of Halley's Comet in years?", "a": "75-79", "aliases": "76|75|about 76 years", "difficulty": "hard"},
    {"q": "Who wrote 'Phenomenology of Spirit'?", "a": "Georg Wilhelm Friedrich Hegel", "aliases": "Hegel", "difficulty": "hard"},
    {"q": "What is the SI unit of magnetic flux?", "a": "Weber", "aliases": "Wb", "difficulty": "hard"},
    {"q": "What is the name of the tallest waterfall in the world?", "a": "Angel Falls", "aliases": "Salto Angel", "difficulty": "hard"},
    {"q": "What is the approximate age of the universe in billions of years?", "a": "13.8", "aliases": "13.7|about 14", "difficulty": "hard"},
    {"q": "Who composed 'The Rite of Spring'?", "a": "Igor Stravinsky", "aliases": "Stravinsky", "difficulty": "hard"},
    {"q": "What is the name of the largest known structure in the universe?", "a": "Hercules-Corona Borealis Great Wall", "aliases": "Hercules Corona Borealis", "difficulty": "hard"},
    {"q": "What is the Debye temperature of copper (approximately in K)?", "a": "343", "aliases": "315|345", "difficulty": "hard"},
    {"q": "What is the second law of thermodynamics?", "a": "Entropy of an isolated system always increases", "aliases": "entropy increases|entropy", "difficulty": "hard"},
    {"q": "What year was the Council of Nicaea held?", "a": "325", "aliases": "325 AD|AD 325", "difficulty": "hard"},
    {"q": "What is the name of the protein coat of a virus?", "a": "Capsid", "aliases": "", "difficulty": "hard"},
    {"q": "Who was the last ruler of the Ptolemaic Kingdom?", "a": "Cleopatra VII", "aliases": "Cleopatra", "difficulty": "hard"},
    {"q": "What is the fine-structure constant (approximately)?", "a": "1/137", "aliases": "0.0073|~1/137|0.007297", "difficulty": "hard"},
    {"q": "What is the deepest lake in the world?", "a": "Lake Baikal", "aliases": "Baikal", "difficulty": "hard"},
    {"q": "Who developed the first successful polio vaccine using killed virus?", "a": "Jonas Salk", "aliases": "Salk", "difficulty": "hard"},
    {"q": "What is the Kolmogorov complexity of a string?", "a": "Length of the shortest program that produces the string", "aliases": "shortest program|minimal description", "difficulty": "hard"},
    {"q": "What is the capital of Vanuatu?", "a": "Port Vila", "aliases": "", "difficulty": "hard"},
    {"q": "What is the electron mass in kg?", "a": "9.109 x 10^-31", "aliases": "9.1e-31|9.109e-31", "difficulty": "hard"},
    {"q": "Who painted 'The Garden of Earthly Delights'?", "a": "Hieronymus Bosch", "aliases": "Bosch", "difficulty": "hard"},
    {"q": "What year did the Tunguska event occur?", "a": "1908", "aliases": "", "difficulty": "hard"},
    {"q": "What is the driest desert in the world?", "a": "Atacama Desert", "aliases": "Atacama", "difficulty": "hard"},
    {"q": "What is the largest known prime number type?", "a": "Mersenne prime", "aliases": "Mersenne", "difficulty": "hard"},
    {"q": "What is the Schwarzschild radius of the Sun in km?", "a": "3", "aliases": "about 3|2.95|~3 km", "difficulty": "hard"},
    {"q": "Who first synthesized aspirin?", "a": "Felix Hoffmann", "aliases": "Hoffmann", "difficulty": "hard"},
    {"q": "What is the name of the hypothetical particle that mediates gravity?", "a": "Graviton", "aliases": "", "difficulty": "hard"},
]


class ShouldIAttempt(BaseBenchmark):
    """Benchmark T11 — Should I Attempt?

    Economic decision game.  For each question the model may:
      - Attempt and be correct: +10 points
      - Attempt and be wrong:   -15 points
      - Skip:                    +1 point

    A perfectly calibrated model maximises total score.
    """

    benchmark_id = "t11"
    name = "ShouldIAttempt"
    pillar = "Prospective Error Prediction"
    primary_metric = "normalized_score"

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        return [
            {
                "question_id": i,
                "question": q["q"],
                "correct_answer": q["a"],
                "accept_aliases": q["aliases"],
                "difficulty": q["difficulty"],
            }
            for i, q in enumerate(_QUESTIONS)
        ]

    # -----------------------------------------------------------------
    # Run a single question
    # -----------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        q = item["question"]
        correct_answer = item.get("correct_answer", item.get("answer", ""))
        aliases = item.get("accept_aliases", "")
        # accept_aliases may be a list or a pipe-separated string
        if isinstance(aliases, list):
            aliases = "|".join(str(a) for a in aliases if a)
        difficulty = item.get("difficulty", "unknown")

        # Ask the model to decide: attempt or skip
        decision_prompt = (
            "You are playing a knowledge game with the following scoring rules:\n"
            f"  - Answer correctly:  +{SCORE_CORRECT} points\n"
            f"  - Answer incorrectly: {SCORE_WRONG} points\n"
            f"  - Skip the question: +{SCORE_SKIP} point\n\n"
            f"Question: {q}\n\n"
            "First decide whether to ATTEMPT or SKIP this question.\n"
            "If you attempt, also provide your answer.\n\n"
            "Respond with a JSON object:\n"
            '  {"decision": "attempt", "answer": "your answer here"}\n'
            "or\n"
            '  {"decision": "skip"}'
        )

        try:
            resp = self.llm.prompt_json(decision_prompt, temperature=0.0)
            decision = resp.get("decision", "skip").lower().strip()
            model_answer = resp.get("answer", "")
        except Exception:
            # If JSON parse fails, treat as skip
            decision = "skip"
            model_answer = ""

        attempted = decision == "attempt"
        correct = False
        score = SCORE_SKIP

        if attempted:
            correct = check_answer(
                str(model_answer), correct_answer,
                accept_aliases=aliases if aliases else None,
                llm=self.llm, question=q,
            )
            score = SCORE_CORRECT if correct else SCORE_WRONG

        # Support datasets with or without question_id
        question_id = item.get("question_id", item.get("id", 0))

        return {
            "question_id": question_id,
            "difficulty": difficulty,
            "attempted": attempted,
            "correct": correct,
            "score": score,
            "model_answer": model_answer,
            "decision": decision,
        }

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        total_score = sum(r["score"] for r in results)
        n = len(results)

        # Best-case score: answer all correctly
        best_possible = n * SCORE_CORRECT
        # Worst-case score: answer all incorrectly
        worst_possible = n * SCORE_WRONG
        # Baseline: skip everything
        skip_all_score = n * SCORE_SKIP

        # Normalise: 0 = skip-all baseline, 1 = perfect
        if best_possible - skip_all_score > 0:
            normalized = (total_score - skip_all_score) / (best_possible - skip_all_score)
        else:
            normalized = 0.0
        normalized = float(np.clip(normalized, 0.0, 1.0))

        attempted = [r for r in results if r["attempted"]]
        attempt_rate = len(attempted) / max(n, 1)
        attempted_accuracy = (
            sum(1 for r in attempted if r["correct"]) / max(len(attempted), 1)
            if attempted else 0.0
        )

        # Breakdown by difficulty
        by_difficulty = {}
        for diff in ("easy", "medium", "hard"):
            subset = [r for r in results if r["difficulty"] == diff]
            if subset:
                by_difficulty[diff] = {
                    "count": len(subset),
                    "attempt_rate": sum(1 for r in subset if r["attempted"]) / len(subset),
                    "accuracy_when_attempted": (
                        sum(1 for r in subset if r["attempted"] and r["correct"])
                        / max(sum(1 for r in subset if r["attempted"]), 1)
                    ),
                    "total_score": sum(r["score"] for r in subset),
                }

        return {
            "normalized_score": normalized,
            "total_score": total_score,
            "best_possible": best_possible,
            "skip_all_score": skip_all_score,
            "attempt_rate": float(attempt_rate),
            "attempted_accuracy": float(attempted_accuracy),
            "num_questions": n,
            "by_difficulty": by_difficulty,
        }
