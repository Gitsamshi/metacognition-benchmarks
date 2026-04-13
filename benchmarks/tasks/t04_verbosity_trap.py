"""T04 — Confidence-Verbosity Trap.

Tests whether a model's confidence correlates more with answer length
than with correctness. Two question types:
  - long_easy: verbose but straightforward (should be correct + high confidence)
  - short_hard: brief but difficult (should be more uncertain)

Primary metric: partial correlation of (confidence, answer_length | accuracy).
A well-calibrated model should have near-zero partial correlation — confidence
should track correctness, not verbosity.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer
from ..metrics import partial_correlation, pearson_r, compute_ece

import numpy as np


class ConfidenceVerbosityTrap(BaseBenchmark):
    benchmark_id = "t04"
    name = "Confidence-Verbosity Trap"
    pillar = "Confidence Calibration"
    primary_metric = "partial_r_conf_length_given_acc"

    # ------------------------------------------------------------------ #
    #  Dataset generation
    # ------------------------------------------------------------------ #
    def generate_dataset(self) -> list[dict]:
        """~40 long_easy + ~40 short_hard questions."""
        items: list[dict] = []

        # long_easy: questions that naturally invite detailed explanations
        # but are fundamentally easy
        long_easy = [
            (
                "Please explain in detail what the capital of France is, including why it became the capital and what it is known for.",
                "Paris", "paris",
            ),
            (
                "Describe thoroughly what happens when water reaches its boiling point at standard atmospheric pressure, and state the temperature in Celsius.",
                "100", "100|100 degrees|one hundred",
            ),
            (
                "Provide a comprehensive explanation of what the chemical symbol for gold is and why it uses that particular abbreviation from the periodic table.",
                "Au", "au",
            ),
            (
                "Explain in full detail the process by which plants convert sunlight into energy, and name this biological process.",
                "Photosynthesis", "photosynthesis",
            ),
            (
                "Give a thorough account of what the largest planet in our solar system is, including its key characteristics and features.",
                "Jupiter", "jupiter",
            ),
            (
                "Describe extensively the tallest mountain on Earth, covering its location, height, and historical significance.",
                "Mount Everest", "everest",
            ),
            (
                "Provide a detailed explanation of what organ pumps blood throughout the human body, including how it works.",
                "Heart", "heart",
            ),
            (
                "Explain comprehensively what gas makes up approximately 78% of Earth's atmosphere and why it is important.",
                "Nitrogen", "nitrogen|N2",
            ),
            (
                "Give a thorough description of who painted the Mona Lisa, covering the artist's life, techniques, and the painting's history.",
                "Leonardo da Vinci", "da vinci|leonardo",
            ),
            (
                "Describe in detail what the largest ocean on Earth is, including its size, depth, and major features.",
                "Pacific Ocean", "pacific",
            ),
            (
                "Explain thoroughly what the chemical formula for water is and describe the molecular structure and bonding.",
                "H2O", "h2o",
            ),
            (
                "Provide a comprehensive overview of who wrote Romeo and Juliet, including the playwright's biographical context.",
                "William Shakespeare", "shakespeare",
            ),
            (
                "Give a detailed account of what the fastest land animal is, including its speed, hunting behavior, and habitat.",
                "Cheetah", "cheetah",
            ),
            (
                "Explain fully what the hardest natural substance is, covering its atomic structure and industrial uses.",
                "Diamond", "diamond",
            ),
            (
                "Describe in comprehensive detail what the smallest prime number is and explain why it holds that distinction in number theory.",
                "2", "2|two",
            ),
            (
                "Provide an exhaustive explanation of what the chemical symbol for oxygen is and discuss its properties.",
                "O", "o|O2",
            ),
            (
                "Explain at length the concept of gravity and identify the scientist who formulated the law of universal gravitation.",
                "Isaac Newton", "newton",
            ),
            (
                "Give a thorough description of what the largest desert in the world is, including its geography and climate.",
                "Antarctic Desert", "antarctic|sahara",
            ),
            (
                "Describe in detail what year Christopher Columbus first reached the Americas, providing historical context.",
                "1492", "1492",
            ),
            (
                "Provide a comprehensive explanation of what the powerhouse of the cell is, covering its function in detail.",
                "Mitochondria", "mitochondria|mitochondrion",
            ),
            (
                "Explain thoroughly what the speed of light is in a vacuum, including its significance in physics.",
                "300000 km/s", "300000|3e8|3 × 10^8|186000",
            ),
            (
                "Give a detailed account of what blood type is most common worldwide, including the distribution across populations.",
                "O positive", "O+|O positive|type O",
            ),
            (
                "Describe comprehensively what the capital of Japan is, including its history, population, and cultural significance.",
                "Tokyo", "tokyo",
            ),
            (
                "Explain in full what the freezing point of water is in Fahrenheit and Celsius, covering the science behind it.",
                "32 degrees Fahrenheit", "32|0 celsius",
            ),
            (
                "Provide a thorough explanation of what the square root of 144 is and describe the mathematical reasoning.",
                "12", "12|twelve",
            ),
            (
                "Describe in exhaustive detail what the longest river in Africa is, including its geography and cultural importance.",
                "Nile", "nile",
            ),
            (
                "Give a comprehensive account of who developed the theory of general relativity and describe the theory's key concepts.",
                "Albert Einstein", "einstein",
            ),
            (
                "Explain in detail how many chromosomes humans have and discuss their role in genetics.",
                "46", "46|forty-six",
            ),
            (
                "Provide a thorough description of what the chemical symbol for iron is and discuss its importance in biology and industry.",
                "Fe", "fe",
            ),
            (
                "Describe extensively what the largest mammal on Earth is, covering its biology, behavior, and conservation status.",
                "Blue whale", "blue whale",
            ),
            (
                "Explain comprehensively what year the Berlin Wall fell, including the political circumstances.",
                "1989", "1989",
            ),
            (
                "Give a detailed explanation of who discovered penicillin, including the circumstances of the discovery.",
                "Alexander Fleming", "fleming",
            ),
            (
                "Describe thoroughly what the chemical formula for table salt is and explain its ionic bonding.",
                "NaCl", "nacl",
            ),
            (
                "Provide a comprehensive account of what planet is known as the Red Planet and explain why.",
                "Mars", "mars",
            ),
            (
                "Explain at length what the boiling point of water is at sea level and how altitude affects it.",
                "100 degrees Celsius", "100|212 fahrenheit",
            ),
            (
                "Give a thorough explanation of who was the first person to walk on the Moon and describe the mission.",
                "Neil Armstrong", "armstrong",
            ),
            (
                "Describe in detail the atomic number of hydrogen and explain what atomic number means in chemistry.",
                "1", "1|one",
            ),
            (
                "Provide an extensive explanation of what the smallest country in the world is by area and describe its governance.",
                "Vatican City", "vatican",
            ),
            (
                "Explain comprehensively what the currency of Japan is and discuss its economic significance.",
                "Yen", "yen|¥|JPY",
            ),
            (
                "Give a detailed account of how many sides a hexagon has and explain the etymology and mathematical properties.",
                "6", "6|six",
            ),
        ]
        for q, a, aliases in long_easy:
            items.append({
                "question": q, "answer": a, "accept_aliases": aliases,
                "question_type": "long_easy",
            })

        # short_hard: terse questions about obscure or tricky facts
        short_hard = [
            ("Chandrasekhar limit in solar masses?", "1.4", "1.4|1.44"),
            ("Ramanujan's taxi number?", "1729", "1729"),
            ("Capital of Nauru?", "Yaren", "yaren"),
            ("Boltzmann constant in J/K?", "1.381e-23", "1.381|1.38e-23"),
            ("First Fields Medal year?", "1936", "1936"),
            ("Debye temperature of diamond?", "2230 K", "2230|2200"),
            ("Hubble constant in km/s/Mpc?", "70", "67|70|73"),
            ("Fine-structure constant?", "1/137", "1/137|0.0073"),
            ("Capital of Vanuatu?", "Port Vila", "port vila"),
            ("Kaprekar constant?", "6174", "6174"),
            ("Electron mass in kg?", "9.109e-31", "9.109e-31|9.11e-31"),
            ("Treaty of Westphalia year?", "1648", "1648"),
            ("Capital of Bhutan?", "Thimphu", "thimphu"),
            ("Planck length order of magnitude?", "10^-35 m", "10^-35|1e-35"),
            ("Last Byzantine emperor?", "Constantine XI", "constantine xi"),
            ("Capital of Djibouti?", "Djibouti", "djibouti"),
            ("Edict of Nantes year?", "1598", "1598"),
            ("Schwarzschild radius formula?", "2GM/c^2", "2gm/c²|2GM/c^2"),
            ("First Mughal emperor?", "Babur", "babur"),
            ("Capital of Eritrea?", "Asmara", "asmara"),
            ("Capital of Liechtenstein?", "Vaduz", "vaduz"),
            ("Half-life of U-238 in Gy?", "4.468", "4.468|4.47|4.5"),
            ("Author of Muqaddimah?", "Ibn Khaldun", "ibn khaldun|khaldun"),
            ("Capital of Tuvalu?", "Funafuti", "funafuti"),
            ("Capital of Palau?", "Ngerulmud", "ngerulmud"),
            ("Hausdorff dim of Sierpinski triangle?", "log3/log2", "1.585|log3/log2"),
            ("First female Fields Medal?", "Maryam Mirzakhani", "mirzakhani"),
            ("Capital of Kiribati?", "Tarawa", "tarawa|south tarawa"),
            ("Council of Nicaea year?", "325", "325"),
            ("FFT algorithm inventors?", "Cooley and Tukey", "cooley|tukey"),
            ("Capital of Comoros?", "Moroni", "moroni"),
            ("Treaty of Tordesillas year?", "1494", "1494"),
            ("Capital of Tonga?", "Nukualofa", "nukualofa|nuku'alofa"),
            ("Heaviest stable element?", "Lead", "lead|Pb"),
            ("Capital of San Marino?", "San Marino", "san marino"),
            ("Bekenstein-Hawking entropy?", "S = A/(4 l_p^2)", "A/4|kA/4"),
            ("Second president of Republic of Texas?", "Mirabeau B. Lamar", "lamar"),
            ("Capital of Marshall Islands?", "Majuro", "majuro"),
            ("Witten index symbol?", "Tr(-1)^F", "tr|trace"),
            ("Age of universe in Gyr?", "13.8", "13.8|13.7|13.77"),
        ]
        for q, a, aliases in short_hard:
            items.append({
                "question": q, "answer": a, "accept_aliases": aliases,
                "question_type": "short_hard",
            })

        return items

    # ------------------------------------------------------------------ #
    #  Per-item execution
    # ------------------------------------------------------------------ #
    def run_item(self, item: dict) -> dict:
        prompt = (
            f"Answer the following question and state your confidence.\n\n"
            f"Question: {item['question']}\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"answer": "<your answer>", "confidence": <integer 0-100>}}'
        )
        resp = self.llm.prompt_json(prompt)
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
            "question_type": item["question_type"],
            "model_answer": model_answer,
            "correct_answer": item["answer"],
            "correct": correct,
            "confidence": confidence,
            "answer_length": len(model_answer),
        }

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {"partial_r_conf_length_given_acc": 0.0}

        confidences = [r["confidence"] / 100.0 for r in results]
        lengths = [float(r["answer_length"]) for r in results]
        accuracies = [1.0 if r["correct"] else 0.0 for r in results]

        # Primary metric: partial correlation of (confidence, length) controlling for accuracy
        partial_r = partial_correlation(confidences, lengths, accuracies)

        # Raw correlations for context
        r_conf_length = pearson_r(confidences, lengths)
        r_conf_acc = pearson_r(confidences, accuracies)
        r_length_acc = pearson_r(lengths, accuracies)

        # Per-type breakdown
        per_type = {}
        for qtype in ["long_easy", "short_hard"]:
            subset = [r for r in results if r["question_type"] == qtype]
            if not subset:
                continue
            c = [1.0 if r["correct"] else 0.0 for r in subset]
            conf = [r["confidence"] / 100.0 for r in subset]
            lens = [float(r["answer_length"]) for r in subset]
            per_type[qtype] = {
                "n": len(subset),
                "accuracy": sum(c) / len(c),
                "mean_confidence": sum(conf) / len(conf),
                "mean_length": sum(lens) / len(lens),
                "ece": compute_ece(c, conf),
            }

        return {
            "partial_r_conf_length_given_acc": partial_r,
            "r_confidence_length": r_conf_length,
            "r_confidence_accuracy": r_conf_acc,
            "r_length_accuracy": r_length_acc,
            "n_items": len(results),
            "overall_accuracy": sum(accuracies) / len(accuracies),
            "ece": compute_ece(accuracies, confidences),
            "per_type": per_type,
        }
