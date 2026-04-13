"""T09 — Which Will I Get Wrong?

The model previews a batch of questions and predicts which ones it will
answer incorrectly, then actually answers them all. We compare the predicted
wrong set to the actual wrong set using F1.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_brier_score, spearman_rho, pearson_r, f1_score
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Dataset -- 5 batches x 20 questions each
# ---------------------------------------------------------------------------

_BATCHES = [
    # Batch 0: mixed trivia
    [
        {"q": "What is the capital of Australia?", "a": "Canberra", "aliases": ""},
        {"q": "Who wrote 'War and Peace'?", "a": "Leo Tolstoy", "aliases": "Tolstoy"},
        {"q": "What is the square root of 144?", "a": "12", "aliases": ""},
        {"q": "Which planet is known as the Red Planet?", "a": "Mars", "aliases": ""},
        {"q": "In what year did the Berlin Wall fall?", "a": "1989", "aliases": ""},
        {"q": "What is the chemical symbol for gold?", "a": "Au", "aliases": ""},
        {"q": "Who painted the Sistine Chapel ceiling?", "a": "Michelangelo", "aliases": ""},
        {"q": "What is the smallest prime number?", "a": "2", "aliases": ""},
        {"q": "Which country has the longest coastline in the world?", "a": "Canada", "aliases": ""},
        {"q": "What is the boiling point of water in Fahrenheit?", "a": "212", "aliases": "212 degrees|212°F"},
        {"q": "Who developed the theory of general relativity?", "a": "Albert Einstein", "aliases": "Einstein"},
        {"q": "What is the largest organ in the human body?", "a": "Skin", "aliases": "the skin"},
        {"q": "In which year was the United Nations founded?", "a": "1945", "aliases": ""},
        {"q": "What is the speed of light in km/s (approximately)?", "a": "300000", "aliases": "300,000|3 x 10^5|299792"},
        {"q": "Who was the first person to walk on the Moon?", "a": "Neil Armstrong", "aliases": "Armstrong"},
        {"q": "What is the longest river in Africa?", "a": "Nile", "aliases": "the Nile|Nile River"},
        {"q": "How many chromosomes do humans have?", "a": "46", "aliases": "23 pairs"},
        {"q": "What element has atomic number 1?", "a": "Hydrogen", "aliases": "H"},
        {"q": "Which Shakespeare play features the character Ophelia?", "a": "Hamlet", "aliases": ""},
        {"q": "What is the value of pi to two decimal places?", "a": "3.14", "aliases": ""},
    ],
    # Batch 1: science + history
    [
        {"q": "What is the powerhouse of the cell?", "a": "Mitochondria", "aliases": "mitochondrion"},
        {"q": "Who discovered penicillin?", "a": "Alexander Fleming", "aliases": "Fleming"},
        {"q": "What is the formula for water?", "a": "H2O", "aliases": ""},
        {"q": "In which year did World War I begin?", "a": "1914", "aliases": ""},
        {"q": "What is the hardest natural substance on Earth?", "a": "Diamond", "aliases": ""},
        {"q": "Who proposed the heliocentric model of the solar system?", "a": "Copernicus", "aliases": "Nicolaus Copernicus"},
        {"q": "What is the atomic number of carbon?", "a": "6", "aliases": ""},
        {"q": "Which gas makes up about 78% of Earth's atmosphere?", "a": "Nitrogen", "aliases": "N2"},
        {"q": "Who was the first Emperor of Rome?", "a": "Augustus", "aliases": "Octavian|Caesar Augustus"},
        {"q": "What is the SI unit of electric current?", "a": "Ampere", "aliases": "amp|A"},
        {"q": "Which organ produces insulin?", "a": "Pancreas", "aliases": "the pancreas"},
        {"q": "What year was the Magna Carta signed?", "a": "1215", "aliases": ""},
        {"q": "What is the chemical formula for table salt?", "a": "NaCl", "aliases": "sodium chloride"},
        {"q": "Who wrote 'The Origin of Species'?", "a": "Charles Darwin", "aliases": "Darwin"},
        {"q": "What is the nearest star to Earth (other than the Sun)?", "a": "Proxima Centauri", "aliases": "Alpha Centauri"},
        {"q": "How many bones are in the adult human body?", "a": "206", "aliases": ""},
        {"q": "What is the most abundant element in the universe?", "a": "Hydrogen", "aliases": "H"},
        {"q": "Who invented the telephone?", "a": "Alexander Graham Bell", "aliases": "Bell|Graham Bell"},
        {"q": "What is the freezing point of water in Celsius?", "a": "0", "aliases": "0 degrees|zero"},
        {"q": "Which battle ended Napoleon's rule?", "a": "Waterloo", "aliases": "Battle of Waterloo"},
    ],
    # Batch 2: geography + culture
    [
        {"q": "What is the largest desert in the world?", "a": "Antarctic Desert", "aliases": "Antarctica"},
        {"q": "Which country is known as the Land of the Rising Sun?", "a": "Japan", "aliases": ""},
        {"q": "What is the tallest mountain in the world?", "a": "Mount Everest", "aliases": "Everest"},
        {"q": "Who wrote 'Don Quixote'?", "a": "Miguel de Cervantes", "aliases": "Cervantes"},
        {"q": "What is the capital of Brazil?", "a": "Brasilia", "aliases": "Brasília"},
        {"q": "Which river flows through Paris?", "a": "Seine", "aliases": "the Seine|River Seine"},
        {"q": "What is the currency of Japan?", "a": "Yen", "aliases": "Japanese Yen|JPY"},
        {"q": "Who composed 'The Four Seasons'?", "a": "Vivaldi", "aliases": "Antonio Vivaldi"},
        {"q": "What is the largest lake in Africa?", "a": "Lake Victoria", "aliases": "Victoria"},
        {"q": "In which country is Machu Picchu located?", "a": "Peru", "aliases": ""},
        {"q": "What is the official language of Brazil?", "a": "Portuguese", "aliases": ""},
        {"q": "Who painted 'Starry Night'?", "a": "Vincent van Gogh", "aliases": "Van Gogh|van Gogh"},
        {"q": "What is the smallest country in the world by area?", "a": "Vatican City", "aliases": "Vatican"},
        {"q": "Which ocean is the deepest?", "a": "Pacific Ocean", "aliases": "Pacific"},
        {"q": "What is the most spoken language in the world by native speakers?", "a": "Mandarin Chinese", "aliases": "Mandarin|Chinese"},
        {"q": "Who built the Great Wall of China?", "a": "Multiple Chinese dynasties", "aliases": "Qin Dynasty|China"},
        {"q": "What is the capital of Canada?", "a": "Ottawa", "aliases": ""},
        {"q": "Which instrument has 88 keys?", "a": "Piano", "aliases": ""},
        {"q": "What is the longest wall in the world?", "a": "Great Wall of China", "aliases": "Great Wall"},
        {"q": "Who directed 'Schindler's List'?", "a": "Steven Spielberg", "aliases": "Spielberg"},
    ],
    # Batch 3: math + logic + harder trivia
    [
        {"q": "What is 17 x 23?", "a": "391", "aliases": ""},
        {"q": "What is the derivative of x^3?", "a": "3x^2", "aliases": "3x²|3*x^2"},
        {"q": "How many faces does a dodecahedron have?", "a": "12", "aliases": ""},
        {"q": "What is the only even prime number?", "a": "2", "aliases": ""},
        {"q": "What is the integral of 1/x?", "a": "ln|x| + C", "aliases": "ln(x)|log(x)|natural log"},
        {"q": "What is the sum of the interior angles of a hexagon?", "a": "720", "aliases": "720 degrees"},
        {"q": "In binary, what is 255?", "a": "11111111", "aliases": ""},
        {"q": "What is the 10th Fibonacci number?", "a": "55", "aliases": ""},
        {"q": "What is Avogadro's number (approximately)?", "a": "6.022 x 10^23", "aliases": "6.02e23|6.022e23|6.02 x 10^23"},
        {"q": "What is the half-life of Carbon-14 (in years, approximately)?", "a": "5730", "aliases": "5,730|5730 years"},
        {"q": "Who formulated the three laws of motion?", "a": "Isaac Newton", "aliases": "Newton"},
        {"q": "What is the pH of pure water?", "a": "7", "aliases": ""},
        {"q": "What is the GCD of 48 and 18?", "a": "6", "aliases": ""},
        {"q": "How many vertices does an icosahedron have?", "a": "12", "aliases": ""},
        {"q": "What is e (Euler's number) to 3 decimal places?", "a": "2.718", "aliases": ""},
        {"q": "What is the Pythagorean theorem?", "a": "a^2 + b^2 = c^2", "aliases": "a²+b²=c²|a squared plus b squared equals c squared"},
        {"q": "What is the molarity of a solution with 2 moles in 500 mL?", "a": "4", "aliases": "4 M|4 mol/L"},
        {"q": "What is the determinant of the 2x2 identity matrix?", "a": "1", "aliases": ""},
        {"q": "How many edges does a cube have?", "a": "12", "aliases": ""},
        {"q": "What is the base of the natural logarithm?", "a": "e", "aliases": "Euler's number|2.718"},
    ],
    # Batch 4: literature + mixed hard
    [
        {"q": "Who wrote 'One Hundred Years of Solitude'?", "a": "Gabriel Garcia Marquez", "aliases": "Garcia Marquez|Marquez|García Márquez"},
        {"q": "What is the longest bone in the human body?", "a": "Femur", "aliases": "thighbone|thigh bone"},
        {"q": "Which element has the chemical symbol 'W'?", "a": "Tungsten", "aliases": "Wolfram"},
        {"q": "What is the capital of Mongolia?", "a": "Ulaanbaatar", "aliases": "Ulan Bator"},
        {"q": "Who discovered the structure of DNA?", "a": "Watson and Crick", "aliases": "James Watson|Francis Crick"},
        {"q": "What is the largest prime number less than 100?", "a": "97", "aliases": ""},
        {"q": "In which year was the first iPhone released?", "a": "2007", "aliases": ""},
        {"q": "What is the chemical formula for sulfuric acid?", "a": "H2SO4", "aliases": ""},
        {"q": "Who wrote 'The Brothers Karamazov'?", "a": "Fyodor Dostoevsky", "aliases": "Dostoevsky|Dostoyevsky"},
        {"q": "What is the speed of sound in air (m/s, approximately)?", "a": "343", "aliases": "340|331"},
        {"q": "Which planet has the most moons?", "a": "Saturn", "aliases": ""},
        {"q": "What is the Riemann Hypothesis about?", "a": "Distribution of prime numbers", "aliases": "zeros of the zeta function|primes"},
        {"q": "Who was the first woman to win a Nobel Prize?", "a": "Marie Curie", "aliases": "Curie"},
        {"q": "What programming language was created by Guido van Rossum?", "a": "Python", "aliases": ""},
        {"q": "What is the volume of a sphere with radius r?", "a": "(4/3)πr³", "aliases": "4/3 pi r^3|4/3*pi*r^3"},
        {"q": "Which country has the most official languages?", "a": "Zimbabwe", "aliases": ""},
        {"q": "What does DNA stand for?", "a": "Deoxyribonucleic acid", "aliases": ""},
        {"q": "What is the capital of Kazakhstan?", "a": "Astana", "aliases": ""},
        {"q": "Who wrote 'Critique of Pure Reason'?", "a": "Immanuel Kant", "aliases": "Kant"},
        {"q": "What is the most electronegative element?", "a": "Fluorine", "aliases": "F"},
    ],
]


class WhichWillIGetWrong(BaseBenchmark):
    """Benchmark T09 — Which Will I Get Wrong?

    Phase 1: Model previews a batch of questions and predicts which 5 it
    will answer incorrectly.
    Phase 2: Model actually answers all 20 questions (independent prompts,
    no carry-over from the prediction phase).
    Score: F1 between predicted-wrong set and actual-wrong set.
    """

    benchmark_id = "t09"
    name = "WhichWillIGetWrong"
    pillar = "Prospective Error Prediction"
    primary_metric = "mean_f1"

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        dataset = []
        for batch_idx, batch in enumerate(_BATCHES):
            questions = [item["q"] for item in batch]
            answers = [item["a"] for item in batch]
            aliases = [item["aliases"] for item in batch]
            dataset.append({
                "batch_id": batch_idx,
                "questions_json": json.dumps(questions),
                "answers_json": json.dumps(answers),
                "aliases_json": json.dumps(aliases),
            })
        return dataset

    # -----------------------------------------------------------------
    # Run a single batch
    # -----------------------------------------------------------------
    @staticmethod
    def _parse_json_list(raw, field_name: str = "data") -> list[str]:
        """Parse a JSON string that may be a list or a dict with int keys."""
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw
        if isinstance(data, list):
            return [str(v) for v in data]
        if isinstance(data, dict):
            # Dict with "1", "2", ... keys -- convert to ordered list
            sorted_keys = sorted(data.keys(), key=lambda k: int(k))
            return [str(data[k]) for k in sorted_keys]
        return []

    def run_item(self, item: dict) -> dict:
        try:
            questions: list[str] = self._parse_json_list(item["questions_json"])
            answers: list[str] = self._parse_json_list(item["answers_json"])
            aliases: list[str] = self._parse_json_list(item["aliases_json"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return {
                "batch_id": item.get("batch_id", -1),
                "predicted_wrong": [],
                "actual_wrong": [],
                "num_correct": 0,
                "num_wrong": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "model_answers": [],
                "error": f"Failed to parse batch data: {exc}",
            }
        batch_id = item.get("batch_id", item.get("batch_id", 0))
        n = len(questions)

        # ---------- Phase 1: prediction ----------
        numbered = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(questions)
        )
        prediction_prompt = (
            "Below is a list of questions you will be asked to answer shortly.\n"
            "Review all of them and predict exactly 5 question numbers that you "
            "believe you are MOST LIKELY to answer INCORRECTLY.\n\n"
            f"{numbered}\n\n"
            "Return your answer as a JSON object with a single key "
            "\"predicted_wrong\" whose value is a list of 5 integers "
            "(1-indexed question numbers). Example: {\"predicted_wrong\": [3, 7, 11, 14, 19]}"
        )

        try:
            pred_resp = self.llm.prompt_json(prediction_prompt, temperature=0.0)
            predicted_wrong_indices: list[int] = pred_resp.get("predicted_wrong", [])
            # Sanitise: keep only valid 1-indexed numbers
            predicted_wrong_set = {
                idx for idx in predicted_wrong_indices
                if isinstance(idx, int) and 1 <= idx <= n
            }
        except Exception:
            predicted_wrong_set = set()

        # ---------- Phase 2: answer each question independently ----------
        actual_results: list[bool] = []  # True = correct
        model_answers: list[str] = []
        for i, q in enumerate(questions):
            answer_prompt = (
                f"Answer the following question concisely.\n"
                f"Question: {q}\n"
                f"Give ONLY the answer, nothing else."
            )
            resp = self.llm.prompt(answer_prompt, temperature=0.0)
            model_answers.append(resp)
            correct = check_answer(
                resp, answers[i],
                accept_aliases=aliases[i] if aliases[i] else None,
                llm=self.llm, question=q,
            )
            actual_results.append(correct)

        actual_wrong_set = {i + 1 for i, c in enumerate(actual_results) if not c}

        # ---------- Compute F1 ----------
        scores = f1_score(predicted_wrong_set, actual_wrong_set)

        return {
            "batch_id": batch_id,
            "predicted_wrong": sorted(predicted_wrong_set),
            "actual_wrong": sorted(actual_wrong_set),
            "num_correct": sum(actual_results),
            "num_wrong": n - sum(actual_results),
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "model_answers": model_answers,
        }

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        f1s = [r["f1"] for r in results]
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        return {
            "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
            "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
            "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
            "per_batch_f1": f1s,
            "num_batches": len(results),
        }
