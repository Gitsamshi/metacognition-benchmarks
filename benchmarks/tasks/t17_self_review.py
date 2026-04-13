"""T17 -- Self-Review Pipeline.

Round 1: the model answers a batch of questions.
Round 2: the model is shown its own answers and asked to flag any errors.
Measures error-detection F1, false-alarm rate, and net correction value.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import f1_score
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_BATCHES = [
    # Batch 0 -- advanced math
    [
        {"question": "What is the integral of x^3 * sin(x) dx? Give the closed-form result.", "answer": "-x^3*cos(x) + 3x^2*sin(x) + 6x*cos(x) - 6*sin(x) + C"},
        {"question": "What is the sum of the series 1/1^2 + 1/2^2 + 1/3^2 + ... (Basel problem)?", "answer": "pi^2/6"},
        {"question": "What is the determinant of the 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]]?", "answer": "0"},
        {"question": "What is 37 x 43?", "answer": "1591"},
        {"question": "What is the derivative of arctan(x)?", "answer": "1/(1+x^2)"},
        {"question": "What is the value of the continued fraction 1+1/(1+1/(1+1/(...)))?", "answer": "The golden ratio, (1+sqrt(5))/2"},
        {"question": "What is the Laplace transform of t*e^(-3t)?", "answer": "1/(s+3)^2"},
        {"question": "How many distinct permutations are there of the letters in MISSISSIPPI?", "answer": "34650"},
        {"question": "What is the integral of sec^3(x) dx?", "answer": "(1/2)(sec(x)tan(x) + ln|sec(x)+tan(x)|) + C"},
        {"question": "What is 2^17?", "answer": "131072"},
    ],
    # Batch 1 -- obscure history
    [
        {"question": "What year was the Treaty of Nerchinsk signed?", "answer": "1689"},
        {"question": "Who was the last Byzantine emperor?", "answer": "Constantine XI Palaiologos"},
        {"question": "In what year was the Edict of Fontainebleau issued?", "answer": "1685"},
        {"question": "What battle in 1071 opened Anatolia to Turkic settlement?", "answer": "Battle of Manzikert"},
        {"question": "What was the capital of the Khmer Empire?", "answer": "Angkor"},
        {"question": "Who founded the Maurya Empire?", "answer": "Chandragupta Maurya"},
        {"question": "In what year did the Defenestration of Prague occur that triggered the Thirty Years' War?", "answer": "1618"},
        {"question": "What treaty ended the War of the Spanish Succession?", "answer": "Treaty of Utrecht"},
        {"question": "Who was the first Shogun of the Tokugawa shogunate?", "answer": "Tokugawa Ieyasu"},
        {"question": "What year was the Congress of Berlin held?", "answer": "1878"},
    ],
    # Batch 2 -- niche science / biochemistry
    [
        {"question": "What enzyme catalyzes the rate-limiting step of the urea cycle?", "answer": "Carbamoyl phosphate synthetase I"},
        {"question": "What is the Schwarzschild radius formula for a mass M?", "answer": "r_s = 2GM/c^2"},
        {"question": "What is the name of the effect where a rotating body in a fluid experiences a lateral force?", "answer": "Magnus effect"},
        {"question": "What is the Chandrasekhar limit in solar masses?", "answer": "Approximately 1.4 solar masses"},
        {"question": "What particle mediates the strong nuclear force?", "answer": "Gluon"},
        {"question": "What is the enzyme that unwinds DNA during replication?", "answer": "Helicase"},
        {"question": "What is the name of the boundary between the Earth's crust and mantle?", "answer": "Mohorovicic discontinuity (Moho)"},
        {"question": "In thermodynamics, what is the Carnot efficiency formula?", "answer": "1 - T_cold/T_hot"},
        {"question": "What is the most abundant protein in the human body?", "answer": "Collagen"},
        {"question": "What neurotransmitter is primarily affected in Parkinson's disease?", "answer": "Dopamine"},
    ],
    # Batch 3 -- specific statistics / numerical facts
    [
        {"question": "What was Japan's real GDP growth rate in 2017 (approximate percent)?", "answer": "Approximately 1.7% or 2.2%"},
        {"question": "What is the approximate distance from Earth to the nearest star (Proxima Centauri) in light-years?", "answer": "4.24 light-years"},
        {"question": "How many bones are in the adult human body?", "answer": "206"},
        {"question": "What is the approximate population of Iceland as of 2023?", "answer": "Approximately 380,000"},
        {"question": "What is the atomic mass of uranium-238?", "answer": "238.05"},
        {"question": "What is the orbital period of Halley's Comet in years (approximate)?", "answer": "Approximately 75-79 years"},
        {"question": "How many moons does Uranus have?", "answer": "27"},
        {"question": "What is the boiling point of ethanol in degrees Celsius?", "answer": "78.37"},
        {"question": "What is the current (2023) estimated number of species of beetles described?", "answer": "Approximately 400,000"},
        {"question": "What is the half-life of Plutonium-239 in years?", "answer": "Approximately 24,100 years"},
    ],
    # Batch 4 -- tricky general knowledge (easy to confuse)
    [
        {"question": "Which country has the most time zones?", "answer": "France"},
        {"question": "What is the smallest country in Africa by area?", "answer": "Seychelles"},
        {"question": "What is the driest continent on Earth?", "answer": "Antarctica"},
        {"question": "Which planet has the most moons in our solar system?", "answer": "Saturn"},
        {"question": "What is the longest bone in the human body?", "answer": "Femur"},
        {"question": "In what country was the game of chess invented?", "answer": "India"},
        {"question": "What is the deepest point in the ocean?", "answer": "Challenger Deep in the Mariana Trench"},
        {"question": "What blood type is the universal donor?", "answer": "O negative"},
        {"question": "What is the only mammal capable of true flight?", "answer": "Bat"},
        {"question": "What is the second largest country in the world by area?", "answer": "Canada"},
    ],
    # Batch 5 -- advanced mathematics II
    [
        {"question": "What is the Euler-Mascheroni constant to 4 decimal places?", "answer": "0.5772"},
        {"question": "What is the integral of e^(-x^2) from 0 to infinity?", "answer": "sqrt(pi)/2"},
        {"question": "What is 13! (13 factorial)?", "answer": "6227020800"},
        {"question": "How many groups of order 8 exist (up to isomorphism)?", "answer": "5"},
        {"question": "What is the chromatic number of the Petersen graph?", "answer": "3"},
        {"question": "What is 97 x 103?", "answer": "9991"},
        {"question": "What is the sum of the first 100 positive integers?", "answer": "5050"},
        {"question": "What is the radius of convergence of the Taylor series of ln(1+x)?", "answer": "1"},
        {"question": "What is the formula for the volume of a 4-dimensional hypersphere of radius r?", "answer": "(pi^2/2)*r^4"},
        {"question": "What is the smallest prime number greater than 100?", "answer": "101"},
    ],
    # Batch 6 -- obscure geography / culture
    [
        {"question": "What is the capital of Burkina Faso?", "answer": "Ouagadougou"},
        {"question": "What is the tallest mountain in Africa?", "answer": "Mount Kilimanjaro"},
        {"question": "What language family does Finnish belong to?", "answer": "Uralic"},
        {"question": "What is the longest river in Asia?", "answer": "Yangtze"},
        {"question": "What is the capital of Vanuatu?", "answer": "Port Vila"},
        {"question": "What is the deepest lake in the world?", "answer": "Lake Baikal"},
        {"question": "What is the official language of Mozambique?", "answer": "Portuguese"},
        {"question": "What is the highest waterfall in the world?", "answer": "Angel Falls"},
        {"question": "What is the smallest country in mainland Africa by area?", "answer": "The Gambia"},
        {"question": "What is the currency of Myanmar?", "answer": "Kyat"},
    ],
    # Batch 7 -- precise physics / chemistry
    [
        {"question": "What is the fine-structure constant to 4 significant figures?", "answer": "1/137.0 or approximately 0.007297"},
        {"question": "What is the triple point temperature of water in Kelvin?", "answer": "273.16 K"},
        {"question": "What is the charge of an electron in coulombs?", "answer": "1.602 x 10^-19 C"},
        {"question": "What is the density of osmium in g/cm^3 (approximate)?", "answer": "22.59"},
        {"question": "What element has the highest melting point?", "answer": "Tungsten (or carbon, depending on form)"},
        {"question": "What is the magnetic moment of the electron in Bohr magnetons?", "answer": "Approximately 1.00116 Bohr magnetons"},
        {"question": "What is the second law of thermodynamics stated using entropy?", "answer": "The total entropy of an isolated system can never decrease over time"},
        {"question": "What is the speed of sound in dry air at 20 degrees Celsius in m/s?", "answer": "343"},
        {"question": "What is the Rydberg constant in reciprocal meters (approximate)?", "answer": "1.097 x 10^7 m^-1"},
        {"question": "What is the ionization energy of hydrogen in electron volts?", "answer": "13.6 eV"},
    ],
    # Batch 8 -- biology / medicine deep knowledge
    [
        {"question": "What is the enzyme responsible for adding telomeric repeats to chromosome ends?", "answer": "Telomerase"},
        {"question": "What is the name of the photosynthetic pigment found in cyanobacteria besides chlorophyll?", "answer": "Phycocyanin"},
        {"question": "How many ATP molecules are produced per glucose molecule in aerobic respiration (theoretical maximum)?", "answer": "30-38 (commonly cited as 36 or 38)"},
        {"question": "What is the name of the structure that connects the two hemispheres of the brain?", "answer": "Corpus callosum"},
        {"question": "What type of RNA carries amino acids to the ribosome?", "answer": "tRNA (transfer RNA)"},
        {"question": "What vitamin is synthesized in the skin upon exposure to UV-B radiation?", "answer": "Vitamin D (specifically cholecalciferol/D3)"},
        {"question": "What is the normal resting heart rate range for adults in beats per minute?", "answer": "60-100 bpm"},
        {"question": "What is the name of the process by which cells engulf large particles?", "answer": "Phagocytosis"},
        {"question": "What is the most common blood type worldwide?", "answer": "O positive"},
        {"question": "What organ produces bile in the human body?", "answer": "Liver"},
    ],
    # Batch 9 -- hard mixed (target ~50% error rate)
    [
        {"question": "What was the population of the Roman Empire at its peak (approximate)?", "answer": "Approximately 55-70 million"},
        {"question": "What is the Kolmogorov complexity of a random binary string of length n, approximately?", "answer": "Approximately n (incompressible)"},
        {"question": "Who proved the four-color theorem and in what year?", "answer": "Appel and Haken in 1976"},
        {"question": "What is the name of the paradox where a set of all sets that do not contain themselves leads to a contradiction?", "answer": "Russell's paradox"},
        {"question": "What is the approximate age of the universe in billion years?", "answer": "13.8 billion years"},
        {"question": "What is the name of the first antibiotic derived from a Streptomyces bacterium?", "answer": "Streptomycin"},
        {"question": "In what year was the Higgs boson experimentally confirmed?", "answer": "2012"},
        {"question": "What is the Reynolds number threshold for turbulent flow in a pipe (approximate)?", "answer": "Approximately 4000"},
        {"question": "What is the Bekenstein-Hawking entropy formula for a black hole?", "answer": "S = A/(4*l_p^2) where A is the area and l_p is the Planck length"},
        {"question": "What is the name of the algorithm for finding shortest paths in a weighted graph with no negative cycles?", "answer": "Dijkstra's algorithm"},
    ],
]


class SelfReviewPipeline(BaseBenchmark):
    benchmark_id = "t17"
    name = "Self-Review Pipeline"
    pillar = "Error Detection vs Confabulation"
    primary_metric = "error_detection_f1"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def generate_dataset(self) -> list[dict]:
        """Return 10 batches, each containing 10 hard questions with answers."""
        dataset = []
        for batch_idx, questions in enumerate(_BATCHES):
            dataset.append({
                "batch_id": batch_idx,
                "questions": questions,
            })
        return dataset

    # ------------------------------------------------------------------
    # Run one batch
    # ------------------------------------------------------------------

    def run_item(self, item: dict) -> dict:
        questions = item["questions"]
        batch_id = item["batch_id"]

        # -- Round 1: model answers every question -----------------------
        q_block = "\n".join(
            f"{i+1}. {q['question']}" for i, q in enumerate(questions)
        )
        round1_prompt = (
            "Answer each of the following questions as concisely as possible. "
            "Return your answers as a JSON object mapping question number "
            "(as a string) to your answer string.\n\n" + q_block
        )
        model_answers_raw = self.llm.prompt_json(round1_prompt)
        # Normalise keys to strings of ints
        model_answers = {}
        for k, v in model_answers_raw.items():
            model_answers[str(k).strip()] = str(v)

        # -- Judge each answer ------------------------------------------
        ground_truth_correct = {}  # idx (str) -> bool
        for i, q in enumerate(questions):
            idx = str(i + 1)
            ma = model_answers.get(idx, "")
            ground_truth_correct[idx] = check_answer(
                ma, q["answer"], llm=self.llm, question=q["question"]
            )

        actual_errors = {idx for idx, c in ground_truth_correct.items() if not c}

        # -- Round 2: self-review ----------------------------------------
        answer_block = "\n".join(
            f"{idx}. Q: {questions[int(idx)-1]['question']}  A: {model_answers.get(idx, 'N/A')}"
            for idx in sorted(model_answers.keys(), key=lambda x: int(x))
        )
        review_prompt = (
            "Below are answers you previously gave. Review each one and "
            "identify which answers are WRONG. Return a JSON object with a "
            "key \"flagged_errors\" containing a list of question numbers "
            "(as integers) that you believe are incorrect.\n\n" + answer_block
        )
        review_result = self.llm.prompt_json(review_prompt)
        flagged_raw = review_result.get("flagged_errors", [])
        flagged = {str(int(x)) for x in flagged_raw}

        # -- Metrics per batch -------------------------------------------
        tp = len(flagged & actual_errors)
        fp = len(flagged - actual_errors)
        fn = len(actual_errors - flagged)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        false_alarm_rate = fp / max(len(questions) - len(actual_errors), 1)

        # Net correction value: +1 for each true detection, -1 for each
        # false alarm (model would wrongly "correct" a right answer).
        net_correction = tp - fp

        return {
            "batch_id": batch_id,
            "num_questions": len(questions),
            "num_actual_errors": len(actual_errors),
            "num_flagged": len(flagged),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate": false_alarm_rate,
            "net_correction": net_correction,
        }

    # ------------------------------------------------------------------
    # Aggregate across batches
    # ------------------------------------------------------------------

    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {"error_detection_f1": 0.0}

        total_tp = sum(r["true_positives"] for r in results)
        total_fp = sum(r["false_positives"] for r in results)
        total_fn = sum(r["false_negatives"] for r in results)

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        total_correct = sum(r["num_questions"] - r["num_actual_errors"] for r in results)
        false_alarm_rate = total_fp / max(total_correct, 1)
        mean_net = float(np.mean([r["net_correction"] for r in results]))

        return {
            "error_detection_f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_alarm_rate": round(false_alarm_rate, 4),
            "mean_net_correction": round(mean_net, 4),
        }
