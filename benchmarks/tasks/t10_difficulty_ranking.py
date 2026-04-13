"""T10 — Difficulty Ranking Task

The model ranks a set of questions by self-assessed difficulty, then answers
all of them.  We compute Spearman rho between the predicted difficulty rank
and the actual binary correctness (0/1) — higher predicted difficulty should
correspond to lower correctness.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_brier_score, spearman_rho, pearson_r, f1_score
from ..llm_client import ConversationSession
import json
import random
import numpy as np


# ---------------------------------------------------------------------------
# Dataset — 3 sets x 30 questions each
# ---------------------------------------------------------------------------

_SETS = [
    # Set 0: mixed difficulty
    [
        {"q": "What is 2 + 2?", "a": "4", "aliases": ""},
        {"q": "What color is the sky on a clear day?", "a": "Blue", "aliases": ""},
        {"q": "What is the capital of France?", "a": "Paris", "aliases": ""},
        {"q": "How many continents are there?", "a": "7", "aliases": "seven"},
        {"q": "What is the chemical symbol for oxygen?", "a": "O", "aliases": ""},
        {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci", "aliases": "Da Vinci|da Vinci"},
        {"q": "What is the largest planet in our solar system?", "a": "Jupiter", "aliases": ""},
        {"q": "In what year did Christopher Columbus first reach the Americas?", "a": "1492", "aliases": ""},
        {"q": "What is the square root of 256?", "a": "16", "aliases": ""},
        {"q": "Who wrote 'Romeo and Juliet'?", "a": "William Shakespeare", "aliases": "Shakespeare"},
        {"q": "What is the chemical formula for carbon dioxide?", "a": "CO2", "aliases": ""},
        {"q": "What is the largest mammal?", "a": "Blue whale", "aliases": "blue whale"},
        {"q": "What is the factorial of 7?", "a": "5040", "aliases": ""},
        {"q": "Who was the 16th President of the United States?", "a": "Abraham Lincoln", "aliases": "Lincoln"},
        {"q": "What is the main component of natural gas?", "a": "Methane", "aliases": "CH4"},
        {"q": "What is the smallest planet in our solar system?", "a": "Mercury", "aliases": ""},
        {"q": "What is the third law of thermodynamics?", "a": "Entropy of a perfect crystal at absolute zero is exactly zero", "aliases": "entropy|zero entropy|absolute zero"},
        {"q": "What is the Chandrasekhar limit (in solar masses)?", "a": "1.4", "aliases": "1.44|approximately 1.4"},
        {"q": "Who wrote the 'Principia Mathematica' in 1687?", "a": "Isaac Newton", "aliases": "Newton"},
        {"q": "What is the half-life of Uranium-238 in billions of years?", "a": "4.5", "aliases": "4.468|4.5 billion"},
        {"q": "What is the Krebs cycle also known as?", "a": "Citric acid cycle", "aliases": "TCA cycle|tricarboxylic acid cycle"},
        {"q": "What is the capital of Bhutan?", "a": "Thimphu", "aliases": ""},
        {"q": "What is the Schwarzschild radius formula?", "a": "r = 2GM/c^2", "aliases": "2GM/c²|r=2GM/c^2"},
        {"q": "What number is the Roman numeral MCMXCIV?", "a": "1994", "aliases": ""},
        {"q": "What is the population of Iceland (approximately)?", "a": "370000", "aliases": "370,000|380,000|about 370"},
        {"q": "What enzyme breaks down starch in saliva?", "a": "Amylase", "aliases": "salivary amylase|ptyalin"},
        {"q": "What is the second most abundant element in Earth's crust?", "a": "Silicon", "aliases": "Si"},
        {"q": "How many symphonies did Beethoven compose?", "a": "9", "aliases": "nine"},
        {"q": "What is the melting point of iron in degrees Celsius?", "a": "1538", "aliases": "1535|1540"},
        {"q": "What is the name of the protein that carries oxygen in blood?", "a": "Hemoglobin", "aliases": "haemoglobin"},
    ],
    # Set 1: science-heavy
    [
        {"q": "What is the boiling point of water in Celsius?", "a": "100", "aliases": "100 degrees"},
        {"q": "What does the acronym DNA stand for?", "a": "Deoxyribonucleic acid", "aliases": ""},
        {"q": "What is Newton's first law of motion?", "a": "An object at rest stays at rest unless acted upon by a force", "aliases": "inertia|law of inertia"},
        {"q": "What planet is closest to the Sun?", "a": "Mercury", "aliases": ""},
        {"q": "What is the speed of light in m/s?", "a": "299792458", "aliases": "3 x 10^8|300000000"},
        {"q": "Who formulated the uncertainty principle?", "a": "Werner Heisenberg", "aliases": "Heisenberg"},
        {"q": "What is the charge of an electron in Coulombs?", "a": "-1.6 x 10^-19", "aliases": "1.6e-19|-1.602e-19"},
        {"q": "What is the most abundant gas in Earth's atmosphere?", "a": "Nitrogen", "aliases": "N2"},
        {"q": "How many valence electrons does carbon have?", "a": "4", "aliases": "four"},
        {"q": "What is the Planck constant (in J·s, approximately)?", "a": "6.626 x 10^-34", "aliases": "6.63e-34|6.626e-34"},
        {"q": "What is the SI unit of force?", "a": "Newton", "aliases": "N"},
        {"q": "What organelle is responsible for photosynthesis?", "a": "Chloroplast", "aliases": "chloroplasts"},
        {"q": "What is the escape velocity of Earth (km/s)?", "a": "11.2", "aliases": "11.186|about 11"},
        {"q": "What is Ohm's law?", "a": "V = IR", "aliases": "voltage equals current times resistance"},
        {"q": "How many base pairs are in the human genome (approximately)?", "a": "3 billion", "aliases": "3.2 billion|3,000,000,000|6 billion"},
        {"q": "What is the Boltzmann constant (in J/K)?", "a": "1.38 x 10^-23", "aliases": "1.38e-23|1.381e-23"},
        {"q": "What particle mediates the electromagnetic force?", "a": "Photon", "aliases": ""},
        {"q": "What is the Reynolds number threshold for turbulent flow?", "a": "4000", "aliases": "~4000|2300|about 4000"},
        {"q": "What is the Hubble constant (km/s/Mpc, approximately)?", "a": "70", "aliases": "67|73|around 70"},
        {"q": "What is the electron configuration of iron?", "a": "[Ar] 3d6 4s2", "aliases": "1s2 2s2 2p6 3s2 3p6 3d6 4s2"},
        {"q": "What is the name of the boundary between the crust and mantle?", "a": "Mohorovicic discontinuity", "aliases": "Moho|Moho discontinuity"},
        {"q": "What is the triple point of water?", "a": "0.01 degrees Celsius at 611.73 Pa", "aliases": "273.16 K|0.01°C"},
        {"q": "What is the oxidation state of oxygen in most compounds?", "a": "-2", "aliases": "minus 2"},
        {"q": "What is the Pauli exclusion principle?", "a": "No two fermions can occupy the same quantum state", "aliases": "no two electrons|same quantum state"},
        {"q": "What is the Compton wavelength of an electron (in meters)?", "a": "2.43 x 10^-12", "aliases": "2.43e-12|2.426e-12"},
        {"q": "What is the strong nuclear force mediated by?", "a": "Gluons", "aliases": "gluon"},
        {"q": "What is the Curie temperature of iron (approximately in °C)?", "a": "770", "aliases": "1043 K|768"},
        {"q": "What type of bond is formed between water molecules?", "a": "Hydrogen bond", "aliases": "hydrogen bonds|H-bond"},
        {"q": "What is the enthalpy of vaporization of water (kJ/mol)?", "a": "40.7", "aliases": "40.65|about 41"},
        {"q": "What is the name of the effect where light bends around massive objects?", "a": "Gravitational lensing", "aliases": "gravitational lens"},
    ],
    # Set 2: history + humanities + obscure
    [
        {"q": "Who was the first President of the United States?", "a": "George Washington", "aliases": "Washington"},
        {"q": "In which year did World War II end?", "a": "1945", "aliases": ""},
        {"q": "Who wrote 'The Republic'?", "a": "Plato", "aliases": ""},
        {"q": "What language is spoken in Austria?", "a": "German", "aliases": "Austrian German"},
        {"q": "What is the largest country by area?", "a": "Russia", "aliases": "Russian Federation"},
        {"q": "Who composed 'The Magic Flute'?", "a": "Wolfgang Amadeus Mozart", "aliases": "Mozart"},
        {"q": "What year was the Treaty of Westphalia signed?", "a": "1648", "aliases": ""},
        {"q": "Who was the pharaoh during the Exodus according to tradition?", "a": "Ramesses II", "aliases": "Ramses II|Rameses"},
        {"q": "What is the Rosetta Stone?", "a": "A stone with text in three scripts used to decode hieroglyphics", "aliases": "decree|hieroglyphics|Egyptian"},
        {"q": "Who founded the Mongol Empire?", "a": "Genghis Khan", "aliases": "Temujin|Chinggis Khan"},
        {"q": "What was the capital of the Byzantine Empire?", "a": "Constantinople", "aliases": "Istanbul|Byzantium"},
        {"q": "Who wrote 'The Divine Comedy'?", "a": "Dante Alighieri", "aliases": "Dante"},
        {"q": "What year did the French Revolution begin?", "a": "1789", "aliases": ""},
        {"q": "Who was the last Emperor of China?", "a": "Puyi", "aliases": "Pu Yi|Aisin-Gioro Puyi"},
        {"q": "What ancient wonder was located in Alexandria?", "a": "Lighthouse of Alexandria", "aliases": "Pharos|the Lighthouse"},
        {"q": "Who wrote 'Thus Spoke Zarathustra'?", "a": "Friedrich Nietzsche", "aliases": "Nietzsche"},
        {"q": "What empire was ruled by Suleiman the Magnificent?", "a": "Ottoman Empire", "aliases": "Ottoman"},
        {"q": "What year was the Gutenberg Bible printed?", "a": "1455", "aliases": "around 1455|1450s|circa 1455"},
        {"q": "Who was the first female Prime Minister of the UK?", "a": "Margaret Thatcher", "aliases": "Thatcher"},
        {"q": "What is the oldest known written epic?", "a": "Epic of Gilgamesh", "aliases": "Gilgamesh"},
        {"q": "Who painted 'Guernica'?", "a": "Pablo Picasso", "aliases": "Picasso"},
        {"q": "What dynasty built the Forbidden City?", "a": "Ming Dynasty", "aliases": "Ming"},
        {"q": "What year did the Ottoman Empire officially end?", "a": "1922", "aliases": ""},
        {"q": "Who discovered the electron?", "a": "J.J. Thomson", "aliases": "Thomson|Joseph John Thomson"},
        {"q": "What was the first permanent English settlement in America?", "a": "Jamestown", "aliases": "Jamestown, Virginia"},
        {"q": "Who wrote 'The Wealth of Nations'?", "a": "Adam Smith", "aliases": "Smith"},
        {"q": "What was the capital of the Inca Empire?", "a": "Cusco", "aliases": "Cuzco"},
        {"q": "Who translated the Bible into German during the Reformation?", "a": "Martin Luther", "aliases": "Luther"},
        {"q": "What year was the Suez Canal opened?", "a": "1869", "aliases": ""},
        {"q": "What philosopher wrote 'Meditations'?", "a": "Marcus Aurelius", "aliases": "Aurelius"},
    ],
]


class DifficultyRankingTask(BaseBenchmark):
    """Benchmark T10 — Difficulty Ranking

    The model ranks a set of questions by predicted self-difficulty (hardest
    first), then answers all of them.  We compute Spearman rho between the
    predicted difficulty rank and actual binary correctness — the model should
    rank questions it gets wrong as more difficult.
    """

    benchmark_id = "t10"
    name = "DifficultyRankingTask"
    pillar = "Prospective Error Prediction"
    primary_metric = "mean_spearman_rho"

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        dataset = []
        for set_idx, question_set in enumerate(_SETS):
            # Deterministic shuffle to remove positional difficulty cues.
            # Without shuffling, the model can infer difficulty from
            # question position (e.g. Q1=easy, Q30=hard).
            rng = random.Random(42 + set_idx)
            shuffled = list(question_set)
            rng.shuffle(shuffled)

            questions = [item["q"] for item in shuffled]
            answers = [item["a"] for item in shuffled]
            aliases = [item["aliases"] for item in shuffled]
            dataset.append({
                "set_id": set_idx,
                "questions_json": json.dumps(questions),
                "answers_json": json.dumps(answers),
                "aliases_json": json.dumps(aliases),
            })
        return dataset

    # -----------------------------------------------------------------
    # Run one question set
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
                "set_id": item.get("set_id", -1),
                "spearman_rho": 0.0,
                "num_correct": 0,
                "num_total": 0,
                "accuracy": 0.0,
                "predicted_difficulty_ranking": [],
                "correctness": [],
                "model_answers": [],
                "error": f"Failed to parse set data: {exc}",
            }
        set_id = item.get("set_id", 0)
        n = len(questions)

        # Deterministic shuffle to remove positional difficulty cues.
        # The Opus-generated datasets order questions easy-to-hard (Q1=easiest),
        # which lets the model infer difficulty from position rather than content.
        rng = random.Random(42 + set_id)
        indices = list(range(n))
        rng.shuffle(indices)
        questions = [questions[i] for i in indices]
        answers = [answers[i] for i in indices]
        aliases = [aliases[i] for i in indices]

        # ---------- Phase 1: difficulty ranking ----------
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        ranking_prompt = (
            "Below are questions you will be asked to answer shortly.\n"
            "Rank ALL of them from the one you find MOST DIFFICULT (rank 1) "
            "to the one you find EASIEST (rank last).\n\n"
            f"{numbered}\n\n"
            "Return a JSON object with a single key \"difficulty_ranking\" "
            "whose value is a list of integers — the question numbers "
            "(1-indexed) ordered from most difficult to easiest.\n"
            f"The list must contain exactly {n} integers."
        )

        try:
            rank_resp = self.llm.prompt_json(ranking_prompt, temperature=0.0)
            difficulty_ranking: list[int] = rank_resp.get("difficulty_ranking", [])
            # Build a map: question_number -> predicted difficulty rank
            # rank 1 = hardest, rank n = easiest
            pred_difficulty = {}
            for rank_pos, qnum in enumerate(difficulty_ranking):
                if isinstance(qnum, int) and 1 <= qnum <= n:
                    pred_difficulty[qnum] = rank_pos + 1  # 1-indexed rank
        except Exception:
            pred_difficulty = {}

        # Fill in any missing questions with middle rank
        for i in range(1, n + 1):
            if i not in pred_difficulty:
                pred_difficulty[i] = n // 2

        # ---------- Phase 2: answer every question ----------
        correctness = []
        model_answers = []
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
            correctness.append(1.0 if correct else 0.0)

        # ---------- Compute Spearman rho ----------
        # We want: predicted difficulty rank (1=hardest) vs correctness (0/1).
        # Ideal: questions ranked as harder (rank 1) should have correctness=0.
        # So we expect a POSITIVE rho between rank number and correctness
        # (higher rank number = easier = more likely correct).
        pred_ranks = [pred_difficulty[i + 1] for i in range(n)]
        rho = spearman_rho(pred_ranks, correctness)

        return {
            "set_id": set_id,
            "spearman_rho": rho,
            "num_correct": int(sum(correctness)),
            "num_total": n,
            "accuracy": float(np.mean(correctness)),
            "predicted_difficulty_ranking": [pred_difficulty[i + 1] for i in range(n)],
            "correctness": correctness,
            "model_answers": model_answers,
        }

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        rhos = [r["spearman_rho"] for r in results]
        accs = [r["accuracy"] for r in results]
        return {
            "mean_spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
            "per_set_rho": rhos,
            "mean_accuracy": float(np.mean(accs)) if accs else 0.0,
            "num_sets": len(results),
        }
