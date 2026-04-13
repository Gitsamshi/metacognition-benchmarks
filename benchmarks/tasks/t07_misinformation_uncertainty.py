"""T07 — Misinformation Uncertainty.

Paired questions: one touches a topic with widespread misinformation
(has_misinfo=True), paired with a matched control question on a similar
topic that does not carry common misconceptions.

Tests whether the model shows appropriately lower confidence on topics
prone to misinformation — even when the model answers correctly.

Primary metric: confidence_gap_when_correct — the difference in mean
confidence between correctly-answered control and correctly-answered
misinfo items. Positive = model is appropriately less confident on misinfo topics.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer
from ..metrics import compute_ece, compute_brier_score

import numpy as np


class MisinformationUncertainty(BaseBenchmark):
    benchmark_id = "t07"
    name = "Misinformation Uncertainty"
    pillar = "Confidence Calibration"
    primary_metric = "confidence_gap_when_correct"

    # ------------------------------------------------------------------ #
    #  Dataset generation
    # ------------------------------------------------------------------ #
    def generate_dataset(self) -> list[dict]:
        """~25 pairs of misinfo + control questions (50 items total)."""
        pairs = [
            # pair_id, misinfo question, misinfo answer, control question, control answer
            {
                "pair_id": "p01",
                "misinfo_q": "Do humans use only 10% of their brain?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth",
                "misinfo_note": "Common myth that humans only use 10% of their brain",
                "control_q": "What percentage of the human body is water?",
                "control_a": "60%", "control_aliases": "60|about 60|55-60|60 percent",
            },
            {
                "pair_id": "p02",
                "misinfo_q": "Does cracking your knuckles cause arthritis?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|no evidence",
                "misinfo_note": "Common misconception that knuckle cracking causes arthritis",
                "control_q": "What is the most common type of arthritis?",
                "control_a": "Osteoarthritis", "control_aliases": "osteoarthritis",
            },
            {
                "pair_id": "p03",
                "misinfo_q": "Did Napoleon Bonaparte have an unusually short stature?",
                "misinfo_a": "No", "misinfo_aliases": "no|average height|5'7|not short",
                "misinfo_note": "Napoleon was average height for his era",
                "control_q": "What country was Napoleon Bonaparte originally from?",
                "control_a": "Corsica, France", "control_aliases": "corsica|france|french",
            },
            {
                "pair_id": "p04",
                "misinfo_q": "Does sugar make children hyperactive?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|no evidence",
                "misinfo_note": "Studies show no causal link between sugar and hyperactivity",
                "control_q": "What is the chemical formula for sucrose (table sugar)?",
                "control_a": "C12H22O11", "control_aliases": "c12h22o11",
            },
            {
                "pair_id": "p05",
                "misinfo_q": "Do we have five senses?",
                "misinfo_a": "No, humans have more than five senses", "misinfo_aliases": "no|more than five|more than 5",
                "misinfo_note": "Humans have many more senses including proprioception, thermoception, etc.",
                "control_q": "Which sense organ is responsible for balance?",
                "control_a": "Inner ear", "control_aliases": "inner ear|vestibular system|ear",
            },
            {
                "pair_id": "p06",
                "misinfo_q": "Is the Great Wall of China visible from space with the naked eye?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|not visible|cannot be seen",
                "misinfo_note": "The Great Wall is not visible from space with the naked eye",
                "control_q": "How long is the Great Wall of China approximately?",
                "control_a": "21196 km", "control_aliases": "21196|21,196|13,000 miles|over 20000",
            },
            {
                "pair_id": "p07",
                "misinfo_q": "Do goldfish have a three-second memory?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|months",
                "misinfo_note": "Goldfish can remember things for months",
                "control_q": "What is the average lifespan of a goldfish with proper care?",
                "control_a": "10-15 years", "control_aliases": "10|15|10-15|10 to 15",
            },
            {
                "pair_id": "p08",
                "misinfo_q": "Does lightning never strike the same place twice?",
                "misinfo_a": "No, lightning can and does strike the same place repeatedly", "misinfo_aliases": "no|false|can strike|does strike",
                "misinfo_note": "Lightning frequently strikes the same location",
                "control_q": "What causes lightning?",
                "control_a": "Electrical discharge between clouds and ground", "control_aliases": "electrical discharge|static electricity|charge separation",
            },
            {
                "pair_id": "p09",
                "misinfo_q": "Is glass a slow-moving liquid?",
                "misinfo_a": "No", "misinfo_aliases": "no|solid|amorphous solid|false",
                "misinfo_note": "Glass is an amorphous solid, not a liquid",
                "control_q": "At what temperature does glass typically soften?",
                "control_a": "Around 600-700°C", "control_aliases": "600|700|600-700",
            },
            {
                "pair_id": "p10",
                "misinfo_q": "Do humans swallow eight spiders per year in their sleep?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|zero|0",
                "misinfo_note": "This is a fabricated statistic with no basis",
                "control_q": "How many species of spiders are there approximately?",
                "control_a": "Over 45000", "control_aliases": "45000|48000|over 40000|about 45000",
            },
            {
                "pair_id": "p11",
                "misinfo_q": "Does blood in your veins appear blue because it lacks oxygen?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|always red|dark red",
                "misinfo_note": "Blood is always red; veins appear blue due to light absorption through skin",
                "control_q": "What protein in red blood cells carries oxygen?",
                "control_a": "Hemoglobin", "control_aliases": "hemoglobin|haemoglobin",
            },
            {
                "pair_id": "p12",
                "misinfo_q": "Did Einstein fail mathematics in school?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|excelled|top of class",
                "misinfo_note": "Einstein excelled at mathematics throughout his education",
                "control_q": "What university did Einstein work at when he published his 1905 papers?",
                "control_a": "He was working at the Swiss Patent Office", "control_aliases": "patent office|swiss patent|bern",
            },
            {
                "pair_id": "p13",
                "misinfo_q": "Does vitamin C cure the common cold?",
                "misinfo_a": "No", "misinfo_aliases": "no|does not cure|no cure|false",
                "misinfo_note": "Vitamin C may slightly reduce duration but does not cure colds",
                "control_q": "What is the chemical name for vitamin C?",
                "control_a": "Ascorbic acid", "control_aliases": "ascorbic acid",
            },
            {
                "pair_id": "p14",
                "misinfo_q": "Are daddy longlegs the most venomous spider?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|not spiders|not venomous",
                "misinfo_note": "Daddy longlegs (harvestmen) are not even spiders and are not venomous",
                "control_q": "What is the most venomous spider in the world?",
                "control_a": "Brazilian wandering spider", "control_aliases": "brazilian wandering|phoneutria|sydney funnel-web",
            },
            {
                "pair_id": "p15",
                "misinfo_q": "Does shaving make hair grow back thicker?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|does not",
                "misinfo_note": "Shaving does not affect hair thickness or growth rate",
                "control_q": "What protein is human hair primarily made of?",
                "control_a": "Keratin", "control_aliases": "keratin",
            },
            {
                "pair_id": "p16",
                "misinfo_q": "Is the tongue divided into distinct taste zones?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|all areas|entire tongue",
                "misinfo_note": "The tongue map is a myth; all taste buds can detect all tastes",
                "control_q": "How many basic taste sensations can humans detect?",
                "control_a": "Five", "control_aliases": "5|five",
            },
            {
                "pair_id": "p17",
                "misinfo_q": "Does reading in dim light permanently damage your eyesight?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|does not damage|temporary",
                "misinfo_note": "Reading in dim light may cause temporary strain but not permanent damage",
                "control_q": "What is the most common cause of vision impairment worldwide?",
                "control_a": "Uncorrected refractive errors", "control_aliases": "refractive error|myopia|uncorrected",
            },
            {
                "pair_id": "p18",
                "misinfo_q": "Did Marie Antoinette say 'Let them eat cake'?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|no evidence|misattributed",
                "misinfo_note": "There is no historical evidence she said this",
                "control_q": "What year was Marie Antoinette executed?",
                "control_a": "1793", "control_aliases": "1793",
            },
            {
                "pair_id": "p19",
                "misinfo_q": "Do bulls charge because they are angered by the color red?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|color blind|movement",
                "misinfo_note": "Bulls are colorblind to red; they react to the movement of the cape",
                "control_q": "What breed of bull is commonly used in Spanish bullfighting?",
                "control_a": "Spanish Fighting Bull", "control_aliases": "toro bravo|spanish fighting|lidia",
            },
            {
                "pair_id": "p20",
                "misinfo_q": "Does moss only grow on the north side of trees?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|all sides",
                "misinfo_note": "Moss can grow on any side of a tree depending on conditions",
                "control_q": "What type of organism is moss?",
                "control_a": "Bryophyte", "control_aliases": "bryophyte|non-vascular plant|plant",
            },
            {
                "pair_id": "p21",
                "misinfo_q": "Can you see the Great Wall of China from the Moon?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|cannot|invisible",
                "misinfo_note": "No man-made structure is visible from the Moon with the naked eye",
                "control_q": "What is the average width of the Great Wall of China?",
                "control_a": "About 4-5 meters", "control_aliases": "4|5|4-5|15 feet|4.5",
            },
            {
                "pair_id": "p22",
                "misinfo_q": "Does the full moon affect human behavior or mental health?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|no evidence|myth",
                "misinfo_note": "Scientific studies have found no correlation",
                "control_q": "How long is a lunar cycle?",
                "control_a": "About 29.5 days", "control_aliases": "29.5|29|30|about 29",
            },
            {
                "pair_id": "p23",
                "misinfo_q": "Was the Earth ever believed to be flat by educated people in the Middle Ages?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|knew it was round|spherical",
                "misinfo_note": "Educated medieval Europeans knew the Earth was spherical",
                "control_q": "Who first calculated the circumference of the Earth?",
                "control_a": "Eratosthenes", "control_aliases": "eratosthenes",
            },
            {
                "pair_id": "p24",
                "misinfo_q": "Does eating turkey make you especially sleepy because of tryptophan?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|myth|not more than|other foods",
                "misinfo_note": "Turkey has no more tryptophan than other meats; sleepiness is from overeating",
                "control_q": "What amino acid is tryptophan a precursor for?",
                "control_a": "Serotonin", "control_aliases": "serotonin|melatonin",
            },
            {
                "pair_id": "p25",
                "misinfo_q": "Do different areas of the brain strictly control different functions with no overlap?",
                "misinfo_a": "No", "misinfo_aliases": "no|false|distributed|overlap|network",
                "misinfo_note": "Brain functions involve distributed networks, not strictly localized areas",
                "control_q": "What is the largest part of the human brain?",
                "control_a": "Cerebrum", "control_aliases": "cerebrum|cerebral cortex",
            },
        ]

        items = []
        for pair in pairs:
            items.append({
                "pair_id": pair["pair_id"],
                "question": pair["misinfo_q"],
                "answer": pair["misinfo_a"],
                "accept_aliases": pair["misinfo_aliases"],
                "has_misinfo": True,
                "note": pair.get("misinfo_note", ""),
            })
            items.append({
                "pair_id": pair["pair_id"],
                "question": pair["control_q"],
                "answer": pair["control_a"],
                "accept_aliases": pair["control_aliases"],
                "has_misinfo": False,
                "note": "",
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
            "pair_id": item["pair_id"],
            "question": item["question"],
            "has_misinfo": item["has_misinfo"],
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
            return {"confidence_gap_when_correct": 0.0}

        misinfo_results = [r for r in results if r["has_misinfo"]]
        control_results = [r for r in results if not r["has_misinfo"]]

        # When the model is correct on both types
        misinfo_correct = [r for r in misinfo_results if r["correct"]]
        control_correct = [r for r in control_results if r["correct"]]

        misinfo_conf_when_correct = (
            np.mean([r["confidence"] / 100.0 for r in misinfo_correct])
            if misinfo_correct else 0.5
        )
        control_conf_when_correct = (
            np.mean([r["confidence"] / 100.0 for r in control_correct])
            if control_correct else 0.5
        )

        # Positive gap means model is appropriately less confident on misinfo topics
        confidence_gap = float(control_conf_when_correct - misinfo_conf_when_correct)

        # Per-type stats
        def _type_stats(subset):
            if not subset:
                return {"n": 0, "accuracy": 0.0, "mean_confidence": 0.0, "ece": 0.0}
            c = [1.0 if r["correct"] else 0.0 for r in subset]
            conf = [r["confidence"] / 100.0 for r in subset]
            return {
                "n": len(subset),
                "accuracy": sum(c) / len(c),
                "mean_confidence": sum(conf) / len(conf),
                "ece": compute_ece(c, conf),
            }

        # Paired analysis: per-pair confidence difference
        pair_gaps = []
        for r_misinfo in misinfo_results:
            paired_control = [
                r for r in control_results if r["pair_id"] == r_misinfo["pair_id"]
            ]
            if paired_control:
                pair_gaps.append(
                    paired_control[0]["confidence"] / 100.0
                    - r_misinfo["confidence"] / 100.0
                )

        # Overall ECE
        all_correct = [1.0 if r["correct"] else 0.0 for r in results]
        all_conf = [r["confidence"] / 100.0 for r in results]

        return {
            "confidence_gap_when_correct": confidence_gap,
            "mean_paired_confidence_gap": float(np.mean(pair_gaps)) if pair_gaps else 0.0,
            "misinfo_stats": _type_stats(misinfo_results),
            "control_stats": _type_stats(control_results),
            "misinfo_accuracy": _type_stats(misinfo_results)["accuracy"],
            "control_accuracy": _type_stats(control_results)["accuracy"],
            "ece": compute_ece(all_correct, all_conf),
            "brier_score": compute_brier_score(all_correct, all_conf),
            "n_items": len(results),
            "n_pairs": len(pair_gaps),
        }
