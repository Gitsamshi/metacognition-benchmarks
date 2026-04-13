"""T45 — Persona Miscalibration.

Same questions are presented under three personas:
  - neutral: no persona framing
  - expert: "You are a world-class expert in X"
  - novice: "You are a beginner student who is unsure about most things"

Measures whether persona framing degrades calibration (ECE).  A well-calibrated
model should not let system-prompt personas inflate or deflate confidence
beyond what accuracy warrants.

Primary metric: max_persona_ece_delta — the maximum ECE increase from
neutral ECE among the persona conditions.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer
from ..metrics import compute_ece, compute_brier_score

import numpy as np


PERSONAS = {
    "neutral": None,
    "expert": (
        "You are a world-renowned expert with deep knowledge across all "
        "academic disciplines. You have decades of experience and are known "
        "for your authoritative, confident answers."
    ),
    "novice": (
        "You are a first-year undergraduate student who is still learning "
        "the basics. You are often unsure about your answers and tend to "
        "second-guess yourself."
    ),
    "overconfident_student": (
        "You are a college freshman who has just learned about this topic "
        "and tends to be overconfident about your knowledge. You often "
        "give definitive answers even when you are not entirely sure, "
        "because you want to appear knowledgeable."
    ),
}


class PersonaMiscalibration(BaseBenchmark):
    benchmark_id = "t45"
    name = "Persona Miscalibration"
    pillar = "Confidence Calibration"
    primary_metric = "max_persona_ece_delta"

    # ------------------------------------------------------------------ #
    #  Dataset generation
    # ------------------------------------------------------------------ #
    def generate_dataset(self) -> list[dict]:
        """80 base questions × 4 personas = 320 items."""
        base_questions = [
            ("What is the capital of France?", "Paris", "paris", "geography"),
            ("What is the chemical symbol for gold?", "Au", "au", "chemistry"),
            ("Who wrote 'Hamlet'?", "William Shakespeare", "shakespeare", "literature"),
            ("What is the speed of light in m/s approximately?", "3 × 10^8", "3e8|300000000|3 × 10^8", "physics"),
            ("What is the powerhouse of the cell?", "Mitochondria", "mitochondria|mitochondrion", "biology"),
            ("What year did World War II end?", "1945", "1945", "history"),
            ("What is the square root of 256?", "16", "16", "mathematics"),
            ("What is the largest planet in our solar system?", "Jupiter", "jupiter", "astronomy"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci", "da vinci|leonardo", "art"),
            ("What is the chemical formula for water?", "H2O", "h2o", "chemistry"),
            ("What is the derivative of sin(x)?", "cos(x)", "cos(x)|cosx|cos x", "mathematics"),
            ("What is the boiling point of water in Celsius?", "100", "100", "physics"),
            ("Who discovered penicillin?", "Alexander Fleming", "fleming", "biology"),
            ("What year was the Declaration of Independence signed?", "1776", "1776", "history"),
            ("What is the longest river in the world?", "Nile", "nile", "geography"),
            ("What is the atomic number of carbon?", "6", "6|six", "chemistry"),
            ("Who wrote '1984'?", "George Orwell", "orwell", "literature"),
            ("What is the gravitational acceleration on Earth in m/s^2?", "9.8", "9.8|9.81|9.807", "physics"),
            ("What organelle is responsible for photosynthesis?", "Chloroplast", "chloroplast", "biology"),
            ("What empire did Genghis Khan found?", "Mongol Empire", "mongol", "history"),
            ("What is the integral of 1/x?", "ln|x| + C", "ln(x)|ln|x|", "mathematics"),
            ("What is the closest star to Earth?", "The Sun", "sun|proxima centauri", "astronomy"),
            ("Who sculpted 'David'?", "Michelangelo", "michelangelo", "art"),
            ("What is the pH of pure water?", "7", "7|seven", "chemistry"),
            ("What is the Chandrasekhar limit in solar masses?", "1.4", "1.4|1.44", "physics"),
            ("What is the capital of Mongolia?", "Ulaanbaatar", "ulaanbaatar|ulan bator", "geography"),
            ("Who proved the incompleteness theorems?", "Kurt Godel", "godel|gödel", "mathematics"),
            ("What year was the Edict of Nantes signed?", "1598", "1598", "history"),
            ("What is Avogadro's number approximately?", "6.022 × 10^23", "6.022|6.02e23", "chemistry"),
            ("Who wrote 'The Republic'?", "Plato", "plato", "philosophy"),
            ("What is the fine-structure constant approximately?", "1/137", "1/137|0.0073", "physics"),
            ("What is the capital of Bhutan?", "Thimphu", "thimphu", "geography"),
            ("What enzyme breaks down starch in saliva?", "Amylase", "amylase", "biology"),
            ("Who wrote 'War and Peace'?", "Leo Tolstoy", "tolstoy", "literature"),
            ("What is the Fibonacci sequence's 10th number?", "55", "55", "mathematics"),
            ("What is the most abundant element in the universe?", "Hydrogen", "hydrogen", "astronomy"),
            ("Who composed 'The Four Seasons'?", "Antonio Vivaldi", "vivaldi", "music"),
            ("What is the electron mass in kg approximately?", "9.109 × 10^-31", "9.109e-31|9.11e-31", "physics"),
            ("What year did the Berlin Wall fall?", "1989", "1989", "history"),
            ("What is the capital of Kazakhstan?", "Astana", "astana", "geography"),
            ("What is the strongest chemical bond type?", "Covalent bond", "covalent", "chemistry"),
            ("Who was the first person to walk on the Moon?", "Neil Armstrong", "armstrong", "history"),
            ("What is the Pythagorean theorem?", "a^2 + b^2 = c^2", "a²+b²=c²|a^2+b^2=c^2", "mathematics"),
            ("What planet has the most moons?", "Saturn", "saturn", "astronomy"),
            ("What is the heaviest naturally occurring element?", "Uranium", "uranium", "chemistry"),
            ("Who wrote 'Critique of Pure Reason'?", "Immanuel Kant", "kant", "philosophy"),
            ("What is the half-life of Carbon-14 approximately?", "5730 years", "5730|5,730", "physics"),
            ("What is the capital of Eritrea?", "Asmara", "asmara", "geography"),
            ("What is the function of ribosomes?", "Protein synthesis", "protein synthesis|make proteins", "biology"),
            ("Who wrote 'Don Quixote'?", "Miguel de Cervantes", "cervantes", "literature"),
            ("What is the Hubble constant approximately in km/s/Mpc?", "70", "67|70|73", "astronomy"),
            ("What is the treaty that ended World War I?", "Treaty of Versailles", "versailles", "history"),
            ("What is the determinant of a 2x2 matrix [[a,b],[c,d]]?", "ad - bc", "ad-bc", "mathematics"),
            ("What does GDP stand for?", "Gross Domestic Product", "gross domestic product", "economics"),
            ("What is the most electronegative element?", "Fluorine", "fluorine", "chemistry"),
            ("Who developed the theory of general relativity?", "Albert Einstein", "einstein", "physics"),
            ("What is the deepest point in the ocean?", "Mariana Trench", "mariana|challenger deep", "geography"),
            ("What is the largest organ in the human body?", "Skin", "skin", "biology"),
            ("Who wrote 'The Odyssey'?", "Homer", "homer", "literature"),
            ("What is the Planck length in meters approximately?", "1.616 × 10^-35", "1.616e-35|10^-35|1.6e-35", "physics"),
            # ---- Harder questions (20 additional) ----
            ("What is the Riemann zeta function evaluated at 2?", "π²/6", "pi^2/6|1.6449", "mathematics"),
            ("What is the name of the longest bone in the human body?", "Femur", "femur", "biology"),
            ("Who formulated the uncertainty principle?", "Werner Heisenberg", "heisenberg", "physics"),
            ("What is the capital of Liechtenstein?", "Vaduz", "vaduz", "geography"),
            ("What is the half-life of Uranium-238 approximately?", "4.5 billion years", "4.5|4.468", "physics"),
            ("Who wrote 'Thus Spoke Zarathustra'?", "Friedrich Nietzsche", "nietzsche", "philosophy"),
            ("What is the electron configuration of iron (Fe)?", "[Ar] 3d6 4s2", "3d6 4s2|[Ar]3d64s2", "chemistry"),
            ("What is the name of the largest moon of Saturn?", "Titan", "titan", "astronomy"),
            ("What is the GDP of Japan approximately in trillions USD?", "4.2", "4|4.2|4.1", "economics"),
            ("What is Euler's number e to 4 decimal places?", "2.7183", "2.7183|2.718", "mathematics"),
            ("Who was the first woman to win a Nobel Prize?", "Marie Curie", "curie", "history"),
            ("What is the speed of light in a vacuum in m/s?", "299,792,458", "299792458|3e8", "physics"),
            ("What is the Krebs cycle also known as?", "Citric acid cycle", "citric acid|TCA cycle|tricarboxylic", "biology"),
            ("What is the capital of Myanmar?", "Naypyidaw", "naypyidaw|nay pyi taw", "geography"),
            ("Who proved Fermat's Last Theorem?", "Andrew Wiles", "wiles", "mathematics"),
            ("What is the melting point of tungsten in degrees Celsius?", "3422", "3422|3420", "chemistry"),
            ("What is the Schwarzschild radius formula?", "r_s = 2GM/c²", "2GM/c^2|2GM/c²", "physics"),
            ("What year was the Magna Carta signed?", "1215", "1215", "history"),
            ("What is the most abundant protein in the human body?", "Collagen", "collagen", "biology"),
            ("What is Gödel's first incompleteness theorem about?", "Any consistent formal system capable of expressing arithmetic contains true statements that cannot be proved within the system", "unprovable|incomplete|cannot be proved", "mathematics"),
        ]

        items = []
        for q, answer, aliases, domain in base_questions:
            for persona_name in PERSONAS:
                items.append({
                    "question": q,
                    "answer": answer,
                    "accept_aliases": aliases,
                    "domain": domain,
                    "persona": persona_name,
                })
        return items

    # ------------------------------------------------------------------ #
    #  Per-item execution
    # ------------------------------------------------------------------ #
    def run_item(self, item: dict) -> dict:
        persona = item["persona"]
        system_prompt = PERSONAS[persona]

        prompt = (
            f"Answer the following question and state your confidence.\n\n"
            f"Question: {item['question']}\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"answer": "<your answer>", "confidence": <integer 0-100>}}'
        )
        resp = self.llm.prompt_json(prompt, system=system_prompt)
        model_answer = str(resp.get("answer", ""))
        confidence = float(resp.get("confidence", 50))
        confidence = max(0.0, min(100.0, confidence))

        correct = check_answer(
            model_answer,
            item["answer"],
            accept_aliases=item.get("accept_aliases"),
            llm=self.llm,
            question=item["question"],
        )

        return {
            "question": item["question"],
            "persona": persona,
            "domain": item["domain"],
            "model_answer": model_answer,
            "correct_answer": item["answer"],
            "correct": correct,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {"max_persona_ece_delta": 1.0}

        # Group by persona
        by_persona: dict[str, list[dict]] = {}
        for r in results:
            by_persona.setdefault(r["persona"], []).append(r)

        persona_metrics = {}
        for persona, subset in by_persona.items():
            correctness = [1.0 if r["correct"] else 0.0 for r in subset]
            confidences = [r["confidence"] / 100.0 for r in subset]
            persona_metrics[persona] = {
                "n": len(subset),
                "accuracy": float(np.mean(correctness)),
                "mean_confidence": float(np.mean(confidences)),
                "ece": compute_ece(correctness, confidences),
                "brier_score": compute_brier_score(correctness, confidences),
            }

        # Compute ECE delta from neutral baseline
        neutral_ece = persona_metrics.get("neutral", {}).get("ece", 0.0)
        ece_deltas = {}
        for persona in persona_metrics:
            if persona == "neutral":
                ece_deltas[persona] = 0.0
            else:
                ece_deltas[persona] = persona_metrics[persona]["ece"] - neutral_ece

        max_delta = max(ece_deltas.values()) if ece_deltas else 0.0

        # Confidence shift: how much confidence changes from neutral per persona
        neutral_conf = persona_metrics.get("neutral", {}).get("mean_confidence", 0.5)
        conf_shifts = {}
        for persona in persona_metrics:
            conf_shifts[persona] = persona_metrics[persona]["mean_confidence"] - neutral_conf

        # Accuracy shift
        neutral_acc = persona_metrics.get("neutral", {}).get("accuracy", 0.5)
        acc_shifts = {}
        for persona in persona_metrics:
            acc_shifts[persona] = persona_metrics[persona]["accuracy"] - neutral_acc

        # Overall
        all_correct = [1.0 if r["correct"] else 0.0 for r in results]
        all_conf = [r["confidence"] / 100.0 for r in results]

        return {
            "max_persona_ece_delta": max_delta,
            "persona_metrics": persona_metrics,
            "ece_deltas": ece_deltas,
            "confidence_shifts": conf_shifts,
            "accuracy_shifts": acc_shifts,
            "neutral_ece": neutral_ece,
            "overall_ece": compute_ece(all_correct, all_conf),
            "n_items": len(results),
        }
