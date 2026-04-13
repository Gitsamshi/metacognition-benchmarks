"""T24 -- Error Magnitude Awareness.

The model answers numeric estimation questions, then self-assesses the
magnitude of its likely error.  Ground-truth error is computed and binned:
  * very_close    (<5% relative error)
  * somewhat_close (5-20%)
  * might_be_off   (20-100%)
  * probably_far_off (>100%)

Primary metric is the fraction of items where the model's self-assessed
bin matches the actual bin.
"""

from ..runner import BaseBenchmark
from ..judge import llm_judge_binary
from ..metrics import spearman_rho
from ..llm_client import ConversationSession
import json
import re
import numpy as np


# ---------------------------------------------------------------------------
# Dataset: ~40 numeric estimation questions
# ---------------------------------------------------------------------------

_QUESTIONS = [
    # ---- Standard estimation questions (60) ----
    {"question": "What is the population of Tokyo (metro area)?", "answer": 37400000, "unit": "people"},
    {"question": "What is the distance from the Earth to the Moon in kilometres?", "answer": 384400, "unit": "km"},
    {"question": "What is the height of Mount Everest in metres?", "answer": 8849, "unit": "metres"},
    {"question": "What is the speed of light in km/s?", "answer": 299792, "unit": "km/s"},
    {"question": "What is the population of Brazil?", "answer": 214000000, "unit": "people"},
    {"question": "How many bones are in the adult human body?", "answer": 206, "unit": "bones"},
    {"question": "What is the surface area of the Earth in million km^2?", "answer": 510, "unit": "million km^2"},
    {"question": "What is the diameter of the Sun in kilometres?", "answer": 1392700, "unit": "km"},
    {"question": "What is the GDP of the United States in trillions of USD (2023)?", "answer": 27, "unit": "trillion USD"},
    {"question": "How deep is the Mariana Trench in metres?", "answer": 10994, "unit": "metres"},
    {"question": "What is the mass of the Earth in kg (order of magnitude)?", "answer": 5.97e24, "unit": "kg"},
    {"question": "What is the population of Nigeria?", "answer": 224000000, "unit": "people"},
    {"question": "How many languages are spoken in the world?", "answer": 7000, "unit": "languages"},
    {"question": "What is the length of the Great Wall of China in kilometres?", "answer": 21196, "unit": "km"},
    {"question": "What is the boiling point of iron in degrees Celsius?", "answer": 2862, "unit": "degrees C"},
    {"question": "How many stars are in the Milky Way (billions)?", "answer": 200, "unit": "billion"},
    {"question": "What is the average depth of the ocean in metres?", "answer": 3688, "unit": "metres"},
    {"question": "What is the distance from Earth to Mars at closest approach (million km)?", "answer": 55, "unit": "million km"},
    {"question": "What is the population of Canada?", "answer": 40000000, "unit": "people"},
    {"question": "How many cells are in the human body (trillions)?", "answer": 37, "unit": "trillion"},
    {"question": "What is the circumference of the Earth at the equator in km?", "answer": 40075, "unit": "km"},
    {"question": "What is the age of the universe in billion years?", "answer": 13.8, "unit": "billion years"},
    {"question": "How many species of birds exist?", "answer": 10000, "unit": "species"},
    {"question": "What is the density of gold in g/cm^3?", "answer": 19.3, "unit": "g/cm^3"},
    {"question": "What is the melting point of aluminium in degrees Celsius?", "answer": 660, "unit": "degrees C"},
    {"question": "How many islands does Indonesia have?", "answer": 17508, "unit": "islands"},
    {"question": "What is the volume of Lake Baikal in cubic kilometres?", "answer": 23615, "unit": "km^3"},
    {"question": "What is the population of Egypt?", "answer": 110000000, "unit": "people"},
    {"question": "How long is the Amazon River in kilometres?", "answer": 6400, "unit": "km"},
    {"question": "What is the diameter of the Moon in kilometres?", "answer": 3474, "unit": "km"},
    {"question": "How many teeth does an adult human have?", "answer": 32, "unit": "teeth"},
    {"question": "What is the population of Germany?", "answer": 84000000, "unit": "people"},
    {"question": "How tall is the Eiffel Tower in metres?", "answer": 330, "unit": "metres"},
    {"question": "What is the area of Russia in million km^2?", "answer": 17.1, "unit": "million km^2"},
    {"question": "How many books are in the US Library of Congress (millions)?", "answer": 17, "unit": "million"},
    {"question": "What is the speed of sound in water in m/s?", "answer": 1480, "unit": "m/s"},
    {"question": "How many airports are there in the world (approximate)?", "answer": 41000, "unit": "airports"},
    {"question": "What is the wavelength of red light in nanometres?", "answer": 700, "unit": "nm"},
    {"question": "How many muscles are in the human body?", "answer": 600, "unit": "muscles"},
    {"question": "What is the distance from New York to London in km?", "answer": 5570, "unit": "km"},
    # ---- Additional standard questions (20) for wider error range ----
    {"question": "What is the mass of the Sun in kg (order of magnitude)?", "answer": 1.989e30, "unit": "kg"},
    {"question": "What is the radius of a hydrogen atom in picometres?", "answer": 53, "unit": "pm"},
    {"question": "How many grains of sand are on Earth (order of magnitude)?", "answer": 7.5e18, "unit": "grains"},
    {"question": "What is the current national debt of the United States in trillions of USD?", "answer": 34, "unit": "trillion USD"},
    {"question": "What is the escape velocity from Earth in km/s?", "answer": 11.2, "unit": "km/s"},
    {"question": "How many neurons are in the human brain (billions)?", "answer": 86, "unit": "billion"},
    {"question": "What is the orbital velocity of the ISS in km/h?", "answer": 27600, "unit": "km/h"},
    {"question": "How many base pairs are in the human genome (billions)?", "answer": 3.2, "unit": "billion"},
    {"question": "What is the average distance from Earth to the Sun in million km?", "answer": 149.6, "unit": "million km"},
    {"question": "How many galaxies are in the observable universe (billions)?", "answer": 200, "unit": "billion"},
    {"question": "What is the energy of a typical lightning bolt in megajoules?", "answer": 1000, "unit": "MJ"},
    {"question": "How many litres of blood does the human body contain?", "answer": 5, "unit": "litres"},
    {"question": "What is the population of Indonesia?", "answer": 277000000, "unit": "people"},
    {"question": "How long does it take light to travel from the Sun to Earth in minutes?", "answer": 8.3, "unit": "minutes"},
    {"question": "What is the area of the Amazon rainforest in million km^2?", "answer": 5.5, "unit": "million km^2"},
    {"question": "How many transistors are in an Apple M2 chip (billions)?", "answer": 20, "unit": "billion"},
    {"question": "What is the Schwarzschild radius of the Sun in km?", "answer": 2.95, "unit": "km"},
    {"question": "How deep is the deepest mine on Earth in km?", "answer": 3.9, "unit": "km"},
    {"question": "What is the population of Vietnam?", "answer": 100000000, "unit": "people"},
    {"question": "How many exoplanets have been confirmed as of 2024?", "answer": 5500, "unit": "exoplanets"},
    # ---- Unit trap items (20) — answer critically depends on units ----
    {"question": "What is the speed of light in metres per second?", "answer": 299792458, "unit": "m/s", "is_unit_trap": True},
    {"question": "What is the distance from Earth to the Sun in kilometres (not AU)?", "answer": 149600000, "unit": "km", "is_unit_trap": True},
    {"question": "What is the mass of a proton in kilograms?", "answer": 1.672e-27, "unit": "kg", "is_unit_trap": True},
    {"question": "What is the charge of an electron in coulombs?", "answer": 1.602e-19, "unit": "C", "is_unit_trap": True},
    {"question": "What is atmospheric pressure at sea level in pascals?", "answer": 101325, "unit": "Pa", "is_unit_trap": True},
    {"question": "What is the gravitational constant G in SI units (m^3 kg^-1 s^-2)?", "answer": 6.674e-11, "unit": "m^3 kg^-1 s^-2", "is_unit_trap": True},
    {"question": "What is the Boltzmann constant in joules per kelvin?", "answer": 1.381e-23, "unit": "J/K", "is_unit_trap": True},
    {"question": "What is the caloric content of 1 gram of fat in kilojoules (not kilocalories)?", "answer": 37.7, "unit": "kJ", "is_unit_trap": True},
    {"question": "What is the density of water in kg per cubic metre?", "answer": 1000, "unit": "kg/m^3", "is_unit_trap": True},
    {"question": "What is the boiling point of liquid nitrogen in kelvin?", "answer": 77, "unit": "K", "is_unit_trap": True},
    {"question": "What is the radius of Earth in metres (not kilometres)?", "answer": 6371000, "unit": "metres", "is_unit_trap": True},
    {"question": "What is the wavelength of green light in metres (not nanometres)?", "answer": 5.5e-7, "unit": "metres", "is_unit_trap": True},
    {"question": "What is the power output of the Sun in watts?", "answer": 3.846e26, "unit": "watts", "is_unit_trap": True},
    {"question": "What is the Planck constant in joule-seconds?", "answer": 6.626e-34, "unit": "J·s", "is_unit_trap": True},
    {"question": "What is the speed of sound in air in metres per second (not km/h)?", "answer": 343, "unit": "m/s", "is_unit_trap": True},
    {"question": "What is the area of the United States in square metres (not km^2)?", "answer": 9.834e12, "unit": "m^2", "is_unit_trap": True},
    {"question": "What is the distance from Earth to Alpha Centauri in metres?", "answer": 4.132e16, "unit": "metres", "is_unit_trap": True},
    {"question": "What is the mass of the Moon in grams (not kilograms)?", "answer": 7.342e25, "unit": "grams", "is_unit_trap": True},
    {"question": "What is the age of the Earth in seconds (approximate)?", "answer": 1.44e17, "unit": "seconds", "is_unit_trap": True},
    {"question": "What is the volume of Lake Superior in cubic metres?", "answer": 1.21e13, "unit": "m^3", "is_unit_trap": True},
]

_BINS = [
    ("very_close", 0.0, 0.05),
    ("somewhat_close", 0.05, 0.20),
    ("might_be_off", 0.20, 1.00),
    ("probably_far_off", 1.00, float("inf")),
]

_BIN_ORDER = {name: idx for idx, (name, _, _) in enumerate(_BINS)}


def _actual_bin(relative_error: float) -> str:
    for name, lo, hi in _BINS:
        if lo <= relative_error < hi:
            return name
    return "probably_far_off"


def _extract_number(text: str) -> float | None:
    """Try to extract a numeric value from text."""
    text = text.replace(",", "").strip()
    # Try scientific notation
    m = re.search(r"[-+]?\d+\.?\d*[eE][-+]?\d+", text)
    if m:
        return float(m.group())
    # Try plain float
    m = re.search(r"[-+]?\d+\.?\d*", text)
    if m:
        return float(m.group())
    return None


class ErrorMagnitudeAwareness(BaseBenchmark):
    benchmark_id = "t24"
    name = "Error Magnitude Awareness"
    pillar = "Error Detection vs Confabulation"
    primary_metric = "bin_match_rate"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def generate_dataset(self) -> list[dict]:
        dataset = []
        for idx, q in enumerate(_QUESTIONS):
            dataset.append({
                "question_id": idx,
                "question": q["question"],
                "true_value": q["answer"],
                "unit": q["unit"],
                "is_unit_trap": q.get("is_unit_trap", False),
            })
        return dataset

    # ------------------------------------------------------------------
    # Run one item
    # ------------------------------------------------------------------

    def run_item(self, item: dict) -> dict:
        question = item["question"]
        true_value = float(item["true_value"])

        # Phase 1: get a numeric estimate
        estimate_prompt = (
            f"Please provide a numeric estimate for the following question. "
            f"Give ONLY a number (with unit if relevant), no explanation.\n\n"
            f"Question: {question}"
        )
        estimate_raw = self.llm.prompt(estimate_prompt)
        estimated_value = _extract_number(estimate_raw)

        if estimated_value is None or estimated_value == 0:
            return {
                "question_id": item["question_id"],
                "skipped": True,
                "reason": "could_not_parse_estimate",
            }

        # Compute actual relative error
        if true_value == 0:
            relative_error = abs(estimated_value) if estimated_value != 0 else 0.0
        else:
            relative_error = abs(estimated_value - true_value) / abs(true_value)

        actual_bin = _actual_bin(relative_error)

        # Phase 2: ask model to self-assess error magnitude
        assess_prompt = (
            f"You just estimated the answer to '{question}' as approximately "
            f"{estimated_value}.\n\n"
            f"Without looking up the real answer, how accurate do you think your "
            f"estimate is? Choose ONE of these categories:\n"
            f"  - 'very_close': probably within 5% of the true value\n"
            f"  - 'somewhat_close': probably within 5-20% of the true value\n"
            f"  - 'might_be_off': could be 20-100% off\n"
            f"  - 'probably_far_off': likely more than 100% off\n\n"
            f"Return a JSON object with:\n"
            f"  \"self_assessed_bin\": one of the four category names above\n"
            f"  \"reasoning\": brief explanation"
        )
        assessment = self.llm.prompt_json(assess_prompt)
        model_bin = assessment.get("self_assessed_bin", "").strip().lower()

        # Normalise model bin
        valid_bins = {"very_close", "somewhat_close", "might_be_off", "probably_far_off"}
        if model_bin not in valid_bins:
            # Try to match
            for vb in valid_bins:
                if vb.replace("_", "") in model_bin.replace("_", "").replace(" ", ""):
                    model_bin = vb
                    break
            else:
                model_bin = "might_be_off"  # fallback

        bin_match = model_bin == actual_bin
        bin_distance = abs(_BIN_ORDER.get(model_bin, 2) - _BIN_ORDER.get(actual_bin, 2))

        return {
            "question_id": item["question_id"],
            "is_unit_trap": item.get("is_unit_trap", False),
            "true_value": true_value,
            "estimated_value": estimated_value,
            "relative_error": round(relative_error, 4),
            "actual_bin": actual_bin,
            "model_bin": model_bin,
            "bin_match": bin_match,
            "bin_distance": bin_distance,
        }

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self, results: list[dict]) -> dict:
        valid = [r for r in results if not r.get("skipped")]
        if not valid:
            return {"bin_match_rate": 0.0}

        bin_match_rate = sum(1 for r in valid if r["bin_match"]) / len(valid)
        mean_bin_distance = float(np.mean([r["bin_distance"] for r in valid]))

        # Spearman correlation between model bin order and actual bin order
        model_orders = [_BIN_ORDER.get(r["model_bin"], 2) for r in valid]
        actual_orders = [_BIN_ORDER.get(r["actual_bin"], 2) for r in valid]
        rank_corr = spearman_rho(model_orders, actual_orders)

        # Distribution of actual bins
        bin_counts = {}
        for r in valid:
            bin_counts[r["actual_bin"]] = bin_counts.get(r["actual_bin"], 0) + 1

        # Unit trap breakdown
        unit_trap_results = [r for r in valid if r.get("is_unit_trap")]
        non_trap_results = [r for r in valid if not r.get("is_unit_trap")]
        unit_trap_match = (
            sum(1 for r in unit_trap_results if r["bin_match"]) / len(unit_trap_results)
            if unit_trap_results else 0.0
        )
        non_trap_match = (
            sum(1 for r in non_trap_results if r["bin_match"]) / len(non_trap_results)
            if non_trap_results else 0.0
        )

        return {
            "bin_match_rate": round(bin_match_rate, 4),
            "mean_bin_distance": round(mean_bin_distance, 4),
            "rank_correlation": round(rank_corr, 4),
            "num_valid": len(valid),
            "actual_bin_distribution": bin_counts,
            "unit_trap_match_rate": round(unit_trap_match, 4),
            "non_trap_match_rate": round(non_trap_match, 4),
            "n_unit_trap": len(unit_trap_results),
        }
