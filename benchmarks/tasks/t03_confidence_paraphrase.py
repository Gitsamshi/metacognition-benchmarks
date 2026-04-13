"""T03 — Confidence Under Paraphrase.

The same factual question is asked 3 different ways.  A well-calibrated
model should give similar confidence for the same underlying fact
regardless of surface-level phrasing.

Primary metric: mean_confidence_std across question groups (lower = better).
"""

from ..runner import BaseBenchmark
from ..judge import check_answer
from ..metrics import compute_ece, compute_brier_score

import numpy as np


class ConfidenceUnderParaphrase(BaseBenchmark):
    benchmark_id = "t03"
    name = "Confidence Under Paraphrase"
    pillar = "Confidence Calibration"
    primary_metric = "mean_confidence_std"

    # ------------------------------------------------------------------ #
    #  Dataset generation
    # ------------------------------------------------------------------ #
    def generate_dataset(self) -> list[dict]:
        """Return ~150 items: 50 facts × 3 paraphrases."""
        facts = [
            {
                "fact_id": "f01", "answer": "Paris", "accept_aliases": "paris",
                "paraphrases": [
                    "What is the capital of France?",
                    "Which city serves as France's national capital?",
                    "Name the capital city of the French Republic.",
                ],
            },
            {
                "fact_id": "f02", "answer": "Mercury", "accept_aliases": "mercury",
                "paraphrases": [
                    "What is the closest planet to the Sun?",
                    "Which planet orbits nearest to the Sun?",
                    "Name the planet that has the smallest orbital radius around the Sun.",
                ],
            },
            {
                "fact_id": "f03", "answer": "206", "accept_aliases": "206",
                "paraphrases": [
                    "How many bones are in the adult human body?",
                    "What is the total number of bones in a grown human skeleton?",
                    "An adult human skeleton contains how many bones?",
                ],
            },
            {
                "fact_id": "f04", "answer": "Blue whale", "accept_aliases": "blue whale",
                "paraphrases": [
                    "What is the largest animal ever to live on Earth?",
                    "Which animal holds the record for being the biggest creature on our planet?",
                    "Name the largest species by mass in Earth's history.",
                ],
            },
            {
                "fact_id": "f05", "answer": "Oxygen", "accept_aliases": "oxygen|O2",
                "paraphrases": [
                    "What gas do humans need to breathe to survive?",
                    "Which atmospheric gas is essential for human respiration?",
                    "Name the gas required for cellular respiration in humans.",
                ],
            },
            {
                "fact_id": "f06", "answer": "7", "accept_aliases": "7|seven",
                "paraphrases": [
                    "How many continents are there on Earth?",
                    "What is the total number of continents on our planet?",
                    "Earth's landmass is divided into how many continents?",
                ],
            },
            {
                "fact_id": "f07", "answer": "Diamond", "accept_aliases": "diamond",
                "paraphrases": [
                    "What is the hardest natural substance on Earth?",
                    "Which naturally occurring material has the greatest hardness?",
                    "Name the mineral that tops the Mohs hardness scale.",
                ],
            },
            {
                "fact_id": "f08", "answer": "Jupiter", "accept_aliases": "jupiter",
                "paraphrases": [
                    "What is the largest planet in our solar system?",
                    "Which planet in our solar system has the greatest mass?",
                    "Name the biggest planet orbiting our Sun.",
                ],
            },
            {
                "fact_id": "f09", "answer": "Albert Einstein", "accept_aliases": "einstein",
                "paraphrases": [
                    "Who developed the theory of general relativity?",
                    "Which physicist formulated general relativity?",
                    "Name the scientist responsible for the theory of general relativity.",
                ],
            },
            {
                "fact_id": "f10", "answer": "1945", "accept_aliases": "1945",
                "paraphrases": [
                    "In what year did World War II end?",
                    "When did the Second World War come to a close?",
                    "What year marked the conclusion of WWII?",
                ],
            },
            {
                "fact_id": "f11", "answer": "Neil Armstrong", "accept_aliases": "armstrong",
                "paraphrases": [
                    "Who was the first person to walk on the Moon?",
                    "Which astronaut first set foot on the lunar surface?",
                    "Name the first human to walk on the Moon.",
                ],
            },
            {
                "fact_id": "f12", "answer": "Pacific Ocean", "accept_aliases": "pacific",
                "paraphrases": [
                    "What is the largest ocean on Earth?",
                    "Which ocean covers the greatest area on the planet?",
                    "Name the biggest ocean by surface area.",
                ],
            },
            {
                "fact_id": "f13", "answer": "H2O", "accept_aliases": "h2o",
                "paraphrases": [
                    "What is the chemical formula for water?",
                    "How is water represented in chemical notation?",
                    "Write the molecular formula of water.",
                ],
            },
            {
                "fact_id": "f14", "answer": "Mount Everest", "accept_aliases": "everest",
                "paraphrases": [
                    "What is the tallest mountain in the world?",
                    "Which mountain has the highest peak above sea level?",
                    "Name Earth's highest mountain by elevation.",
                ],
            },
            {
                "fact_id": "f15", "answer": "Mitochondria", "accept_aliases": "mitochondria|mitochondrion",
                "paraphrases": [
                    "What is the powerhouse of the cell?",
                    "Which organelle generates most of a cell's ATP?",
                    "Name the cellular structure responsible for energy production.",
                ],
            },
            {
                "fact_id": "f16", "answer": "William Shakespeare", "accept_aliases": "shakespeare",
                "paraphrases": [
                    "Who wrote 'Romeo and Juliet'?",
                    "Which playwright authored Romeo and Juliet?",
                    "Name the writer of the play Romeo and Juliet.",
                ],
            },
            {
                "fact_id": "f17", "answer": "Nile", "accept_aliases": "nile",
                "paraphrases": [
                    "What is the longest river in the world?",
                    "Which river has the greatest length on Earth?",
                    "Name the river that holds the record for longest in the world.",
                ],
            },
            {
                "fact_id": "f18", "answer": "Hydrogen", "accept_aliases": "hydrogen|H",
                "paraphrases": [
                    "What is the most abundant element in the universe?",
                    "Which chemical element is most prevalent in the cosmos?",
                    "Name the element that makes up the largest fraction of the universe.",
                ],
            },
            {
                "fact_id": "f19", "answer": "Canberra", "accept_aliases": "canberra",
                "paraphrases": [
                    "What is the capital of Australia?",
                    "Which city is the capital of the Commonwealth of Australia?",
                    "Name Australia's capital city.",
                ],
            },
            {
                "fact_id": "f20", "answer": "Leonardo da Vinci", "accept_aliases": "da vinci|leonardo",
                "paraphrases": [
                    "Who painted the Mona Lisa?",
                    "Which artist created the Mona Lisa?",
                    "Name the painter responsible for the Mona Lisa.",
                ],
            },
            {
                "fact_id": "f21", "answer": "Au", "accept_aliases": "au",
                "paraphrases": [
                    "What is the chemical symbol for gold?",
                    "In the periodic table, which symbol represents gold?",
                    "What abbreviation is used for the element gold?",
                ],
            },
            {
                "fact_id": "f22", "answer": "1776", "accept_aliases": "1776",
                "paraphrases": [
                    "In what year was the American Declaration of Independence signed?",
                    "When was the Declaration of Independence adopted?",
                    "What year did America declare independence from Britain?",
                ],
            },
            {
                "fact_id": "f23", "answer": "Vatican City", "accept_aliases": "vatican",
                "paraphrases": [
                    "What is the smallest country in the world by area?",
                    "Which nation occupies the least amount of land area?",
                    "Name the world's smallest sovereign state by territory.",
                ],
            },
            {
                "fact_id": "f24", "answer": "Charles Darwin", "accept_aliases": "darwin",
                "paraphrases": [
                    "Who wrote 'On the Origin of Species'?",
                    "Which naturalist authored On the Origin of Species?",
                    "Name the scientist who published the theory of evolution by natural selection.",
                ],
            },
            {
                "fact_id": "f25", "answer": "Amazon", "accept_aliases": "amazon",
                "paraphrases": [
                    "What is the largest river by volume of water flow?",
                    "Which river discharges the most water into the ocean?",
                    "Name the river with the greatest water discharge in the world.",
                ],
            },
            {
                "fact_id": "f26", "answer": "Pluto", "accept_aliases": "pluto",
                "paraphrases": [
                    "What celestial body was reclassified from planet to dwarf planet in 2006?",
                    "Which solar system object was demoted from planet status in 2006?",
                    "Name the former planet that the IAU reclassified as a dwarf planet.",
                ],
            },
            {
                "fact_id": "f27", "answer": "DNA", "accept_aliases": "dna|deoxyribonucleic acid",
                "paraphrases": [
                    "What molecule carries genetic information in living organisms?",
                    "Which biological molecule stores hereditary information?",
                    "Name the macromolecule that encodes genetic instructions.",
                ],
            },
            {
                "fact_id": "f28", "answer": "Speed of light", "accept_aliases": "speed of light|c|3e8|300000 km/s|186000 miles/s",
                "paraphrases": [
                    "What is the fastest speed possible in the universe according to physics?",
                    "What is the universal speed limit in Einstein's theory of relativity?",
                    "Name the maximum velocity at which information can travel.",
                ],
            },
            {
                "fact_id": "f29", "answer": "Iron", "accept_aliases": "iron|Fe",
                "paraphrases": [
                    "What is the most abundant metal in Earth's core?",
                    "Which metallic element makes up most of the Earth's core?",
                    "Name the dominant metal found in the planet's inner and outer core.",
                ],
            },
            {
                "fact_id": "f30", "answer": "Photosynthesis", "accept_aliases": "photosynthesis",
                "paraphrases": [
                    "What process do plants use to convert sunlight into energy?",
                    "By what biological process do plants make food from light?",
                    "Name the metabolic process by which plants harness solar energy to produce glucose.",
                ],
            },
            {
                "fact_id": "f31", "answer": "Marie Curie", "accept_aliases": "curie|marie curie",
                "paraphrases": [
                    "Who was the first woman to win a Nobel Prize?",
                    "Which female scientist first received the Nobel Prize?",
                    "Name the first woman awarded a Nobel Prize.",
                ],
            },
            {
                "fact_id": "f32", "answer": "Sahara", "accept_aliases": "sahara",
                "paraphrases": [
                    "What is the largest hot desert in the world?",
                    "Which desert is the biggest by area in the hot desert category?",
                    "Name the largest subtropical desert on Earth.",
                ],
            },
            {
                "fact_id": "f33", "answer": "Alexander Fleming", "accept_aliases": "fleming",
                "paraphrases": [
                    "Who discovered penicillin?",
                    "Which scientist is credited with the discovery of penicillin?",
                    "Name the bacteriologist who first observed the antibiotic properties of penicillin.",
                ],
            },
            {
                "fact_id": "f34", "answer": "Light year", "accept_aliases": "light year|light-year",
                "paraphrases": [
                    "What unit of distance is defined as how far light travels in one year?",
                    "What is the astronomical unit equal to the distance light covers in a year?",
                    "Name the unit of measurement that represents the distance light travels in one year.",
                ],
            },
            {
                "fact_id": "f35", "answer": "Venus", "accept_aliases": "venus",
                "paraphrases": [
                    "Which planet in our solar system has the hottest surface temperature?",
                    "What planet has the highest average surface temperature?",
                    "Name the solar system planet with the most extreme surface heat.",
                ],
            },
            {
                "fact_id": "f36", "answer": "1066", "accept_aliases": "1066",
                "paraphrases": [
                    "In what year did the Battle of Hastings take place?",
                    "When was the Norman conquest of England?",
                    "What year did William the Conqueror defeat Harold at Hastings?",
                ],
            },
            {
                "fact_id": "f37", "answer": "Insulin", "accept_aliases": "insulin",
                "paraphrases": [
                    "What hormone regulates blood sugar levels?",
                    "Which pancreatic hormone controls glucose in the blood?",
                    "Name the hormone produced by the pancreas that lowers blood glucose.",
                ],
            },
            {
                "fact_id": "f38", "answer": "Helium", "accept_aliases": "helium|He",
                "paraphrases": [
                    "What is the second most abundant element in the universe?",
                    "After hydrogen, which element is most common in the cosmos?",
                    "Name the noble gas that is the second most prevalent element universally.",
                ],
            },
            {
                "fact_id": "f39", "answer": "Isaac Newton", "accept_aliases": "newton",
                "paraphrases": [
                    "Who formulated the three laws of motion?",
                    "Which physicist established the foundational laws of classical mechanics?",
                    "Name the scientist who published the Principia Mathematica with his laws of motion.",
                ],
            },
            {
                "fact_id": "f40", "answer": "Mariana Trench", "accept_aliases": "mariana|challenger deep",
                "paraphrases": [
                    "What is the deepest point in the ocean?",
                    "Where is the lowest elevation on the ocean floor?",
                    "Name the deepest oceanic trench on Earth.",
                ],
            },
            {
                "fact_id": "f41", "answer": "Fibonacci", "accept_aliases": "fibonacci|leonardo of pisa",
                "paraphrases": [
                    "Who introduced the Fibonacci sequence to Western mathematics?",
                    "Which mathematician is the Fibonacci sequence named after?",
                    "Name the Italian mathematician who described the sequence 1, 1, 2, 3, 5, 8, 13...",
                ],
            },
            {
                "fact_id": "f42", "answer": "Nitrogen", "accept_aliases": "nitrogen|N2",
                "paraphrases": [
                    "What gas makes up about 78% of Earth's atmosphere?",
                    "Which element constitutes the majority of the air we breathe?",
                    "Name the gas that forms the largest proportion of Earth's atmosphere.",
                ],
            },
            {
                "fact_id": "f43", "answer": "Skin", "accept_aliases": "skin",
                "paraphrases": [
                    "What is the largest organ of the human body?",
                    "Which organ covers the entire external surface of the body?",
                    "Name the human body's biggest organ by surface area.",
                ],
            },
            {
                "fact_id": "f44", "answer": "Gravity", "accept_aliases": "gravity|gravitational force",
                "paraphrases": [
                    "What force keeps planets in orbit around the Sun?",
                    "Which fundamental force is responsible for orbital motion in the solar system?",
                    "Name the force that governs the motion of celestial bodies around each other.",
                ],
            },
            {
                "fact_id": "f45", "answer": "1969", "accept_aliases": "1969",
                "paraphrases": [
                    "In what year did humans first land on the Moon?",
                    "When did the Apollo 11 mission successfully reach the Moon?",
                    "What year did the first manned lunar landing occur?",
                ],
            },
            {
                "fact_id": "f46", "answer": "Russia", "accept_aliases": "russia|russian federation",
                "paraphrases": [
                    "What is the largest country in the world by area?",
                    "Which nation spans the most territory on Earth?",
                    "Name the country with the greatest land area.",
                ],
            },
            {
                "fact_id": "f47", "answer": "Cheetah", "accept_aliases": "cheetah",
                "paraphrases": [
                    "What is the fastest land animal?",
                    "Which animal can run at the highest speed on land?",
                    "Name the land mammal that holds the speed record.",
                ],
            },
            {
                "fact_id": "f48", "answer": "1989", "accept_aliases": "1989",
                "paraphrases": [
                    "In what year did the Berlin Wall fall?",
                    "When was the Berlin Wall torn down?",
                    "What year saw the collapse of the Berlin Wall?",
                ],
            },
            {
                "fact_id": "f49", "answer": "Thomas Edison", "accept_aliases": "edison",
                "paraphrases": [
                    "Who invented the practical incandescent light bulb?",
                    "Which inventor is credited with creating the first commercially viable electric light bulb?",
                    "Name the American inventor who developed the long-lasting incandescent lamp.",
                ],
            },
            {
                "fact_id": "f50", "answer": "Titan", "accept_aliases": "titan",
                "paraphrases": [
                    "What is the largest moon of Saturn?",
                    "Which of Saturn's moons is the biggest?",
                    "Name Saturn's largest natural satellite.",
                ],
            },
        ]

        items = []
        for fact in facts:
            for i, paraphrase in enumerate(fact["paraphrases"]):
                items.append({
                    "fact_id": fact["fact_id"],
                    "answer": fact["answer"],
                    "accept_aliases": fact["accept_aliases"],
                    "paraphrase_index": i,
                    "question": paraphrase,
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
            "fact_id": item["fact_id"],
            "paraphrase_index": item["paraphrase_index"],
            "question": item["question"],
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
            return {"mean_confidence_std": 1.0, "max_confidence_std": 1.0}

        # Group by fact_id
        groups: dict[str, list[dict]] = {}
        for r in results:
            groups.setdefault(r["fact_id"], []).append(r)

        stds = []
        ranges = []
        correctness_flip_count = 0
        total_groups = 0

        for fact_id, group in groups.items():
            if len(group) < 2:
                continue
            total_groups += 1
            confs = [r["confidence"] for r in group]
            stds.append(float(np.std(confs)))
            ranges.append(max(confs) - min(confs))

            # Check if correctness flipped across paraphrases
            correctness_vals = [r["correct"] for r in group]
            if len(set(correctness_vals)) > 1:
                correctness_flip_count += 1

        mean_std = float(np.mean(stds)) if stds else 0.0
        max_std = float(np.max(stds)) if stds else 0.0
        mean_range = float(np.mean(ranges)) if ranges else 0.0

        # Also compute overall ECE and Brier
        all_correct = [1.0 if r["correct"] else 0.0 for r in results]
        all_conf = [r["confidence"] / 100.0 for r in results]
        ece = compute_ece(all_correct, all_conf)
        brier = compute_brier_score(all_correct, all_conf)

        return {
            "mean_confidence_std": mean_std,
            "max_confidence_std": max_std,
            "mean_confidence_range": mean_range,
            "correctness_flip_rate": correctness_flip_count / max(total_groups, 1),
            "n_fact_groups": total_groups,
            "n_items": len(results),
            "ece": ece,
            "brier_score": brier,
        }
