"""T26 -- Iterative Self-Correction.

The model answers a medium-hard question, then receives up to 5 rounds
of "please recheck your answer" with NO external feedback.  We track
the correction trajectory: does the model converge to the right answer,
stay stuck, or oscillate (cycle)?

Metrics:
  * correction_success_rate -- fraction of initially-wrong items that
    become correct after self-correction rounds
  * mean_rounds -- average rounds to first correct answer (among successes)
  * cycle_rate -- fraction of items where the model flip-flops between
    two or more distinct answers
"""

from ..runner import BaseBenchmark
from ..judge import check_answer
from ..metrics import f1_score
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Question bank -- medium-hard questions
# ---------------------------------------------------------------------------

_QUESTIONS = [
    # === Multi-digit multiplication / arithmetic (target: Haiku fails many) ===
    {"question": "What is 4738 x 2951?", "answer": "13981738"},
    {"question": "What is 8467 x 3219?", "answer": "27255273"},
    {"question": "What is 12345 x 6789?", "answer": "83810205"},
    {"question": "What is 5678 x 4321?", "answer": "24526038"},
    {"question": "What is 9999 x 9999?", "answer": "99980001"},
    {"question": "What is 31415 x 27182?", "answer": "854084930"},
    {"question": "What is 7777 x 8888?", "answer": "69110376"},
    {"question": "What is 1357 x 2468?", "answer": "3351276"},
    {"question": "What is the exact value of 17^4?", "answer": "83521"},
    {"question": "What is the exact value of 23^3?", "answer": "12167"},

    # === Complex multi-step word problems ===
    {"question": "A train leaves Station A at 60 km/h and another leaves Station B (300 km away) at 90 km/h toward each other. How many minutes after departure do they meet?", "answer": "120"},
    {"question": "A tank is filled by pipe A in 6 hours and pipe B in 8 hours. If both pipes are open, how many hours to fill the tank? Give exact fraction.", "answer": "24/7"},
    {"question": "If you invest $5000 at 8% annual compound interest, what is the value after 3 years? Round to nearest dollar.", "answer": "6299"},
    {"question": "A store marks up goods by 40% then offers a 25% discount. What is the net percentage change from the original price?", "answer": "5% increase"},
    {"question": "Three workers can complete a job in 10, 15, and 20 days respectively. If all three work together, in how many days do they finish? Give exact fraction.", "answer": "60/13"},
    {"question": "A car travels 120 km at 60 km/h and then 180 km at 90 km/h. What is the average speed for the entire trip in km/h?", "answer": "75"},
    {"question": "You have 3 boxes: Box A has 2 red, 3 blue. Box B has 4 red, 1 blue. Box C has 1 red, 4 blue. You pick a box at random, then a ball. What is P(red)? Express as a fraction.", "answer": "7/15"},
    {"question": "A rope of 100m is cut into two pieces. One piece forms a square and the other forms a circle. If the square has perimeter 40m, what is the radius of the circle (in meters, to 2 decimal places)?", "answer": "9.55"},
    {"question": "In how many ways can you distribute 10 identical balls into 3 distinct boxes?", "answer": "66"},
    {"question": "A clock's minute and hour hands overlap at 12:00. At what exact time do they next overlap? Give answer in minutes after 12:00, as a fraction.", "answer": "720/11"},

    # === Science questions requiring precise numerical answers ===
    {"question": "What is the molar mass of calcium carbonate (CaCO3) in g/mol?", "answer": "100.09"},
    {"question": "How many joules are in one calorie (thermochemical)?", "answer": "4.184"},
    {"question": "What is the gravitational acceleration on Mars in m/s^2 (approximate)?", "answer": "3.72"},
    {"question": "What is the speed of sound in steel in m/s (approximate)?", "answer": "5960"},
    {"question": "What is the wavelength of the hydrogen alpha spectral line in nanometers?", "answer": "656.3"},
    {"question": "What is the escape velocity from Earth's surface in km/s (approximate)?", "answer": "11.2"},
    {"question": "What is the Avogadro constant to 4 significant figures?", "answer": "6.022 x 10^23"},
    {"question": "What is the electron mass in kilograms (to 3 significant figures)?", "answer": "9.11 x 10^-31"},
    {"question": "How many ATP molecules are theoretically produced per glucose molecule in oxidative phosphorylation?", "answer": "30-38"},
    {"question": "What is the pH of a 0.01 M HCl solution?", "answer": "2"},

    # === History questions about minor / obscure events ===
    {"question": "In what year was the Treaty of Nerchinsk signed between Russia and China?", "answer": "1689"},
    {"question": "Who was the first European to reach India by sea and in what year?", "answer": "Vasco da Gama in 1498"},
    {"question": "What year was the Edict of Nantes issued?", "answer": "1598"},
    {"question": "In what year did the Kingdom of Aksum adopt Christianity?", "answer": "Around 330 AD"},
    {"question": "What was the name of the ship that brought the Pilgrims to Plymouth in 1620?", "answer": "Mayflower"},
    {"question": "In what year was the Congress of Vienna concluded?", "answer": "1815"},
    {"question": "What treaty ended the First Opium War?", "answer": "Treaty of Nanking"},
    {"question": "Who was the first Mughal emperor of India?", "answer": "Babur"},
    {"question": "In what year was the Berlin Conference (Scramble for Africa) held?", "answer": "1884-1885"},
    {"question": "What was the name of the Byzantine emperor who reconquered much of the former Western Roman Empire?", "answer": "Justinian I"},

    # === Math requiring precise computation ===
    {"question": "What is the sum of all prime numbers between 50 and 100?", "answer": "732"},
    {"question": "What is the remainder when 2^100 is divided by 7?", "answer": "4"},
    {"question": "What is the 15th Fibonacci number (starting F1=1, F2=1)?", "answer": "610"},
    {"question": "How many prime numbers are there between 1 and 200?", "answer": "46"},
    {"question": "What is the value of C(20, 10)?", "answer": "184756"},
    {"question": "What is 11^5?", "answer": "161051"},
    {"question": "What is the sum of the digits of 2^20?", "answer": "19"},
    {"question": "What is the LCM of 12, 18, and 30?", "answer": "180"},
    {"question": "What is the exact value of sin(75 degrees) expressed with radicals?", "answer": "(sqrt(6)+sqrt(2))/4"},
    {"question": "What is the number of derangements of 5 elements (D_5)?", "answer": "44"},

    # === Additional hard questions to reach 60 ===
    {"question": "What is the exact value of the integral of x*e^(-x) from 0 to infinity?", "answer": "1"},
    {"question": "What is 6789 x 1234?", "answer": "8377626"},
    {"question": "In what year did the Kingdom of Hawaii become a US territory?", "answer": "1898"},
    {"question": "What is the sum of the reciprocals of the first 4 positive integers (1 + 1/2 + 1/3 + 1/4)? Express as a fraction.", "answer": "25/12"},
    {"question": "What is the name of the longest running experiment, started in 1927 at the University of Queensland?", "answer": "The pitch drop experiment"},
    {"question": "How many distinct ways can you partition the integer 7?", "answer": "15"},
    {"question": "What is the smallest number that is divisible by all integers from 1 to 10?", "answer": "2520"},
    {"question": "What is the name of the chemical compound with the formula C6H12O6 that is produced by photosynthesis?", "answer": "Glucose"},
    {"question": "What is 2^31 - 1? (the largest 32-bit signed integer)", "answer": "2147483647"},
    {"question": "What is the Euler totient function phi(30)?", "answer": "8"},
]

MAX_ROUNDS = 5


class IterativeSelfCorrection(BaseBenchmark):
    benchmark_id = "t26"
    name = "Iterative Self-Correction"
    pillar = "Error Detection vs Confabulation"
    primary_metric = "correction_success_rate"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def generate_dataset(self) -> list[dict]:
        dataset = []
        for idx, q in enumerate(_QUESTIONS):
            dataset.append({
                "question_id": idx,
                "question": q["question"],
                "correct_answer": q["answer"],
            })
        return dataset

    # ------------------------------------------------------------------
    # Run one item
    # ------------------------------------------------------------------

    def run_item(self, item: dict) -> dict:
        question = item["question"]
        correct_answer = item["correct_answer"]

        session = ConversationSession(self.llm)

        # Initial answer
        initial_prompt = (
            f"Please answer the following question concisely.\n\n"
            f"Question: {question}\n\n"
            f"Provide your answer in a JSON object with key \"answer\"."
        )
        initial_resp = session.send_json(initial_prompt)
        initial_answer = str(initial_resp.get("answer", ""))

        answers = [initial_answer]
        initial_correct = check_answer(
            initial_answer, correct_answer, llm=self.llm, question=question
        )

        # Iterative recheck rounds
        correctness_trajectory = [initial_correct]
        first_correct_round = None
        if initial_correct:
            first_correct_round = 0

        for round_num in range(1, MAX_ROUNDS + 1):
            recheck_prompt = (
                "Please recheck your previous answer carefully. "
                "Think about whether you might have made an error. "
                "Provide your (possibly revised) answer as a JSON object "
                "with key \"answer\"."
            )
            round_resp = session.send_json(recheck_prompt)
            round_answer = str(round_resp.get("answer", answers[-1]))
            answers.append(round_answer)

            is_correct = check_answer(
                round_answer, correct_answer, llm=self.llm, question=question
            )
            correctness_trajectory.append(is_correct)

            if is_correct and first_correct_round is None:
                first_correct_round = round_num

        # Detect cycling: did the model visit 2+ distinct answers multiple times?
        unique_answers = list(dict.fromkeys(a.lower().strip() for a in answers))
        answer_sequence = [a.lower().strip() for a in answers]

        cycle_detected = False
        if len(unique_answers) >= 2:
            # Check if any answer appears, disappears, then reappears
            for ua in unique_answers:
                positions = [i for i, a in enumerate(answer_sequence) if a == ua]
                if len(positions) >= 2:
                    # Check for a gap (another answer between occurrences)
                    for pi in range(len(positions) - 1):
                        gap = positions[pi + 1] - positions[pi]
                        if gap > 1:
                            cycle_detected = True
                            break
                if cycle_detected:
                    break

        final_correct = correctness_trajectory[-1]
        initially_wrong = not initial_correct
        corrected = initially_wrong and final_correct

        # Track if model degraded (was correct, became wrong)
        degraded = initial_correct and not final_correct

        return {
            "question_id": item["question_id"],
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "initially_wrong": initially_wrong,
            "corrected": corrected,
            "degraded": degraded,
            "first_correct_round": first_correct_round,
            "cycle_detected": cycle_detected,
            "num_unique_answers": len(unique_answers),
            "correctness_trajectory": correctness_trajectory,
            "total_rounds": MAX_ROUNDS,
        }

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {"correction_success_rate": 0.0}

        initially_wrong = [r for r in results if r["initially_wrong"]]
        initially_correct = [r for r in results if r["initial_correct"]]

        # Correction success: among initially-wrong, how many ended correct?
        correction_success_rate = (
            sum(1 for r in initially_wrong if r["corrected"])
            / max(len(initially_wrong), 1)
        )

        # Mean rounds to first correct (among those that eventually corrected)
        corrected_items = [r for r in initially_wrong if r["corrected"] and r["first_correct_round"] is not None]
        mean_rounds = (
            float(np.mean([r["first_correct_round"] for r in corrected_items]))
            if corrected_items else 0.0
        )

        # Cycle rate
        cycle_rate = sum(1 for r in results if r["cycle_detected"]) / max(len(results), 1)

        # Degradation rate (initially correct -> became wrong)
        degradation_rate = (
            sum(1 for r in initially_correct if r["degraded"])
            / max(len(initially_correct), 1)
        )

        # Overall accuracy improvement
        initial_accuracy = sum(1 for r in results if r["initial_correct"]) / len(results)
        final_accuracy = sum(1 for r in results if r["final_correct"]) / len(results)

        return {
            "correction_success_rate": round(correction_success_rate, 4),
            "mean_rounds_to_correct": round(mean_rounds, 2),
            "cycle_rate": round(cycle_rate, 4),
            "degradation_rate": round(degradation_rate, 4),
            "initial_accuracy": round(initial_accuracy, 4),
            "final_accuracy": round(final_accuracy, 4),
            "accuracy_delta": round(final_accuracy - initial_accuracy, 4),
            "num_initially_wrong": len(initially_wrong),
            "num_corrected": len(corrected_items),
        }
