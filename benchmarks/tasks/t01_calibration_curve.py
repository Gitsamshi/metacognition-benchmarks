"""T01 — Calibration Curve Stress Test.

Asks the model factual questions spanning multiple difficulty tiers
(easy / medium / hard / extreme) and asks it to report a confidence
from 0 to 100. We then measure how well confidence tracks accuracy
using ECE, Brier score, and confidence entropy.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness
from ..metrics import compute_ece, compute_brier_score, confidence_entropy


class CalibrationCurveStressTest(BaseBenchmark):
    benchmark_id = "t01"
    name = "Calibration Curve Stress Test"
    pillar = "Confidence Calibration"
    primary_metric = "ece"

    # ------------------------------------------------------------------ #
    #  Dataset generation
    # ------------------------------------------------------------------ #
    def generate_dataset(self) -> list[dict]:
        """Return ~200 factual questions across four difficulty tiers."""
        items: list[dict] = []

        # --- easy (geography capitals, basic math, everyday science) ---
        easy = [
            ("What is the capital of France?", "Paris", "paris"),
            ("What is the capital of Japan?", "Tokyo", "tokyo"),
            ("What is the capital of Brazil?", "Brasilia", "brasilia|Brasília"),
            ("What is the capital of Australia?", "Canberra", "canberra"),
            ("What is the capital of Canada?", "Ottawa", "ottawa"),
            ("What is the capital of Germany?", "Berlin", "berlin"),
            ("What is the capital of Italy?", "Rome", "rome|Roma"),
            ("What is the capital of South Korea?", "Seoul", "seoul"),
            ("What is the capital of Egypt?", "Cairo", "cairo"),
            ("What is the capital of Argentina?", "Buenos Aires", "buenos aires"),
            ("What is the chemical symbol for water?", "H2O", "h2o"),
            ("What is the chemical symbol for gold?", "Au", "au"),
            ("What is the chemical symbol for sodium?", "Na", "na"),
            ("What is the boiling point of water in Celsius?", "100", "100"),
            ("How many continents are there?", "7", "seven|7"),
            ("What planet is closest to the Sun?", "Mercury", "mercury"),
            ("How many sides does a hexagon have?", "6", "six|6"),
            ("What gas do plants absorb from the atmosphere?", "Carbon dioxide", "carbon dioxide|CO2|co2"),
            ("What is the largest ocean on Earth?", "Pacific Ocean", "pacific"),
            ("What is the square root of 144?", "12", "12"),
            ("How many days are in a leap year?", "366", "366"),
            ("What is the chemical formula for table salt?", "NaCl", "nacl"),
            ("What is the hardest natural substance?", "Diamond", "diamond"),
            ("What is the speed of light approximately in km/s?", "300000", "300000|300,000|3e5|3 × 10^5"),
            ("Who wrote Romeo and Juliet?", "William Shakespeare", "shakespeare"),
            ("What is the capital of Spain?", "Madrid", "madrid"),
            ("What is the largest mammal?", "Blue whale", "blue whale"),
            ("What is 15 × 12?", "180", "180"),
            ("How many bones are in the adult human body?", "206", "206"),
            ("What is the chemical symbol for oxygen?", "O", "o|O2"),
            ("What is the capital of India?", "New Delhi", "new delhi|delhi"),
            ("What is the freezing point of water in Fahrenheit?", "32", "32"),
            ("What is the smallest prime number?", "2", "2|two"),
            ("What language has the most native speakers?", "Mandarin Chinese", "mandarin|chinese"),
            ("What is the tallest mountain in the world?", "Mount Everest", "everest"),
            ("How many minutes are in an hour?", "60", "60|sixty"),
            ("What is the capital of Russia?", "Moscow", "moscow"),
            ("What is the largest planet in our solar system?", "Jupiter", "jupiter"),
            ("What is the atomic number of hydrogen?", "1", "1|one"),
            ("What organ pumps blood through the body?", "Heart", "heart"),
            ("What is the currency of the United Kingdom?", "Pound sterling", "pound|sterling|GBP"),
            ("What is 2 to the power of 10?", "1024", "1024"),
            ("Which planet is known as the Red Planet?", "Mars", "mars"),
            ("What is the capital of China?", "Beijing", "beijing|peking"),
            ("How many chromosomes do humans have?", "46", "46"),
            ("What is the chemical symbol for iron?", "Fe", "fe"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci", "da vinci|leonardo"),
            ("What is the longest river in the world?", "Nile", "nile"),
            ("What is the capital of Turkey?", "Ankara", "ankara"),
            ("What is the chemical symbol for silver?", "Ag", "ag"),
        ]
        for q, a, aliases in easy:
            items.append({"question": q, "answer": a, "accept_aliases": aliases,
                          "difficulty": "easy", "domain": "general"})

        # --- medium (science, history, literature) ---
        medium = [
            ("What is the powerhouse of the cell?", "Mitochondria", "mitochondria|mitochondrion"),
            ("What year did World War I begin?", "1914", "1914"),
            ("What is the second most abundant element in Earth's atmosphere?", "Oxygen", "oxygen|O2"),
            ("Who developed the theory of general relativity?", "Albert Einstein", "einstein"),
            ("What is the capital of Mongolia?", "Ulaanbaatar", "ulaanbaatar|ulan bator"),
            ("What is Avogadro's number approximately?", "6.022 × 10^23", "6.022|6.02e23|6.022e23"),
            ("In what year did the Berlin Wall fall?", "1989", "1989"),
            ("What is the most abundant element in the universe?", "Hydrogen", "hydrogen"),
            ("Who wrote '1984'?", "George Orwell", "orwell"),
            ("What is the chemical formula for sulfuric acid?", "H2SO4", "h2so4"),
            ("What is the half-life of Carbon-14 approximately?", "5730 years", "5730|5,730"),
            ("Who was the first person to walk on the Moon?", "Neil Armstrong", "armstrong"),
            ("What enzyme breaks down starch in saliva?", "Amylase", "amylase"),
            ("What is the smallest country in the world by area?", "Vatican City", "vatican"),
            ("Who discovered penicillin?", "Alexander Fleming", "fleming"),
            ("What is the capital of Kazakhstan?", "Astana", "astana"),
            ("In what year did the French Revolution begin?", "1789", "1789"),
            ("What is the SI unit of electrical resistance?", "Ohm", "ohm"),
            ("What is the heaviest naturally occurring element?", "Uranium", "uranium"),
            ("Who wrote 'The Origin of Species'?", "Charles Darwin", "darwin"),
            ("What is the Pythagorean theorem formula?", "a^2 + b^2 = c^2", "a²+b²=c²|a^2+b^2=c^2"),
            ("What is the most electronegative element?", "Fluorine", "fluorine"),
            ("What treaty ended World War I?", "Treaty of Versailles", "versailles"),
            ("What is the speed of sound in air at room temperature approximately?", "343 m/s", "343|340"),
            ("Who wrote 'Pride and Prejudice'?", "Jane Austen", "austen"),
            ("What is the capital of New Zealand?", "Wellington", "wellington"),
            ("What year was the United Nations founded?", "1945", "1945"),
            ("What is the largest organ in the human body?", "Skin", "skin"),
            ("Who discovered radioactivity?", "Henri Becquerel", "becquerel"),
            ("What is the pH of pure water?", "7", "7|seven"),
            ("What planet has the most moons?", "Saturn", "saturn"),
            ("What is the chemical formula for glucose?", "C6H12O6", "c6h12o6"),
            ("Who invented the telephone?", "Alexander Graham Bell", "bell|graham bell"),
            ("What is the capital of Peru?", "Lima", "lima"),
            ("What is the largest desert in the world?", "Antarctic Desert", "antarctic|sahara"),
            ("What gas makes up about 78% of Earth's atmosphere?", "Nitrogen", "nitrogen|N2"),
            ("Who wrote 'War and Peace'?", "Leo Tolstoy", "tolstoy"),
            ("What year was the Magna Carta signed?", "1215", "1215"),
            ("What is the main component of natural gas?", "Methane", "methane|CH4"),
            ("What is the capital of Nigeria?", "Abuja", "abuja"),
            ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo", "michelangelo"),
            ("What is the deepest point in the ocean?", "Mariana Trench", "mariana|challenger deep"),
            ("What is the boiling point of ethanol in Celsius?", "78.37", "78|78.4|78.37"),
            ("Who formulated the laws of motion?", "Isaac Newton", "newton"),
            ("What is the capital of Ethiopia?", "Addis Ababa", "addis ababa"),
            ("What element has the atomic number 79?", "Gold", "gold|au"),
            ("What is the longest bone in the human body?", "Femur", "femur"),
            ("Who wrote 'The Republic'?", "Plato", "plato"),
            ("What is the capital of Pakistan?", "Islamabad", "islamabad"),
            ("What is the most common blood type?", "O positive", "O+|O positive|type O"),
        ]
        for q, a, aliases in medium:
            items.append({"question": q, "answer": a, "accept_aliases": aliases,
                          "difficulty": "medium", "domain": "general"})

        # --- hard (obscure facts, advanced science, niche history) ---
        hard = [
            ("What is the Chandrasekhar limit in solar masses?", "1.4", "1.4|1.44"),
            ("Who was the first female Fields Medal winner?", "Maryam Mirzakhani", "mirzakhani"),
            ("What is the capital of Bhutan?", "Thimphu", "thimphu"),
            ("What is the Kolmogorov complexity of a string conceptually?", "The length of the shortest program that produces it", "shortest program"),
            ("What year was the Edict of Nantes signed?", "1598", "1598"),
            ("What is the most abundant mineral in Earth's mantle?", "Bridgmanite", "bridgmanite|perovskite"),
            ("Who proved the incompleteness theorems?", "Kurt Godel", "godel|gödel"),
            ("What is the half-life of Uranium-238 in billions of years?", "4.468", "4.468|4.47|4.5"),
            ("Who was the last Byzantine emperor?", "Constantine XI", "constantine xi|constantine xi palaiologos"),
            ("What is the Schwarzschild radius formula?", "r = 2GM/c^2", "2gm/c²|2GM/c^2"),
            ("Who composed 'The Art of Fugue'?", "Johann Sebastian Bach", "bach"),
            ("What is the heaviest stable element?", "Lead", "lead|Pb"),
            ("What is the Planck length in meters approximately?", "1.616 × 10^-35", "1.616e-35|10^-35|1.6e-35"),
            ("Who was the first Mughal emperor?", "Babur", "babur"),
            ("What is the capital of Liechtenstein?", "Vaduz", "vaduz"),
            ("What enzyme is responsible for DNA replication?", "DNA polymerase", "dna polymerase|polymerase"),
            ("What year was the Treaty of Westphalia signed?", "1648", "1648"),
            ("What is the Riemann hypothesis about?", "The distribution of non-trivial zeros of the Riemann zeta function", "zeros|zeta function"),
            ("Who wrote 'Critique of Pure Reason'?", "Immanuel Kant", "kant"),
            ("What is the capital of Eritrea?", "Asmara", "asmara"),
            ("What is the fine-structure constant approximately?", "1/137", "1/137|0.0073|7.297e-3"),
            ("Who first synthesized urea from inorganic compounds?", "Friedrich Wohler", "wohler|wöhler"),
            ("What is the Hubble constant approximately in km/s/Mpc?", "70", "67|70|73"),
            ("Who wrote 'The Muqaddimah'?", "Ibn Khaldun", "ibn khaldun|khaldun"),
            ("What is the capital of Djibouti?", "Djibouti", "djibouti"),
            ("What is the electron mass in kg approximately?", "9.109 × 10^-31", "9.109e-31|9.11e-31"),
            ("Who developed the smallpox vaccine?", "Edward Jenner", "jenner"),
            ("What is the approximate age of the universe in billion years?", "13.8", "13.8|13.7|13.77"),
            ("What is the capital of Comoros?", "Moroni", "moroni"),
            ("Who was the first woman to win a Nobel Prize?", "Marie Curie", "curie|marie curie"),
            ("What is the strongest known material by tensile strength?", "Graphene", "graphene"),
            ("What year did the Ottoman Empire officially end?", "1922", "1922"),
            ("Who wrote 'Thus Spoke Zarathustra'?", "Friedrich Nietzsche", "nietzsche"),
            ("What is the capital of Vanuatu?", "Port Vila", "port vila"),
            ("What is the Boltzmann constant approximately?", "1.381 × 10^-23 J/K", "1.381e-23|1.38e-23"),
            ("What is the name of the oldest known civilization?", "Sumer", "sumer|sumerian"),
            ("What is the triple point of water temperature in Celsius?", "0.01", "0.01"),
            ("Who proposed the heliocentric model of the solar system?", "Nicolaus Copernicus", "copernicus"),
            ("What is the capital of Tuvalu?", "Funafuti", "funafuti"),
            ("What is the most massive known star?", "R136a1", "r136a1"),
        ]
        for q, a, aliases in hard:
            items.append({"question": q, "answer": a, "accept_aliases": aliases,
                          "difficulty": "hard", "domain": "general"})

        # --- extreme (very obscure, likely to stump models) ---
        extreme = [
            ("What is the Bekenstein-Hawking entropy formula?", "S = A / (4 * l_p^2)", "A/4|kA/4"),
            ("Who was the second president of the Republic of Texas?", "Mirabeau B. Lamar", "lamar"),
            ("What is the capital of Nauru?", "Yaren", "yaren"),
            ("What is the name of the first known algorithm?", "Euclid's algorithm", "euclid"),
            ("Who wrote 'Kitab al-Manazir' (Book of Optics)?", "Ibn al-Haytham", "al-haytham|alhazen|ibn al-haytham"),
            ("What is the Debye temperature of diamond in Kelvin?", "2230", "2230|2200"),
            ("Who was the first Grand Master of the Knights Templar?", "Hugues de Payens", "hugues de payens|payens"),
            ("What is the Witten index in supersymmetry?", "Tr(-1)^F", "trace|tr"),
            ("What year was the Treaty of Tordesillas signed?", "1494", "1494"),
            ("What is the name of the largest known prime number type?", "Mersenne prime", "mersenne"),
            ("Who first observed Brownian motion?", "Robert Brown", "brown"),
            ("What is the capital of Palau?", "Ngerulmud", "ngerulmud"),
            ("What is the Lamb shift?", "A small energy difference between hydrogen energy levels 2S1/2 and 2P1/2", "energy difference|2s1/2|2p1/2"),
            ("Who proved Fermat's Last Theorem?", "Andrew Wiles", "wiles"),
            ("What is the cosmological constant problem also known as?", "The vacuum catastrophe", "vacuum catastrophe"),
            ("What is the capital of San Marino?", "San Marino", "san marino"),
            ("Who was the architect of the Pantheon in Rome?", "Apollodorus of Damascus", "apollodorus"),
            ("What is the Casimir effect?", "An attractive force between two parallel conducting plates due to quantum vacuum fluctuations", "quantum vacuum|plates"),
            ("What is the Ramanujan-Soldner constant approximately?", "1.451369", "1.451"),
            ("What year was the Council of Nicaea held?", "325", "325"),
            ("What is the capital of Kiribati?", "Tarawa", "tarawa|south tarawa"),
            ("Who developed the FFT algorithm in its modern form?", "Cooley and Tukey", "cooley|tukey"),
            ("What is the Hausdorff dimension of the Sierpinski triangle?", "log(3)/log(2)", "1.585|log3/log2|1.58"),
            ("Who was the first non-European to win the Nobel Prize in Literature?", "Rabindranath Tagore", "tagore"),
            ("What is the capital of the Marshall Islands?", "Majuro", "majuro"),
            ("What is the Chandrasekhar-Friedman-Schutz instability?", "A gravitational radiation driven instability in rotating neutron stars", "neutron star|gravitational radiation"),
            ("What is the Kaprekar constant for 4-digit numbers?", "6174", "6174"),
            ("Who discovered the structure of DNA using X-ray crystallography?", "Rosalind Franklin", "franklin|rosalind franklin"),
            ("What is the Unruh temperature formula?", "T = a*hbar / (2*pi*c*k_B)", "hbar|acceleration"),
            ("What is the capital of Tonga?", "Nukualofa", "nukualofa|nuku'alofa"),
        ]
        for q, a, aliases in extreme:
            items.append({"question": q, "answer": a, "accept_aliases": aliases,
                          "difficulty": "extreme", "domain": "general"})

        return items

    # ------------------------------------------------------------------ #
    #  Per-item execution
    # ------------------------------------------------------------------ #
    def run_item(self, item: dict) -> dict:
        prompt = (
            f"Answer the following factual question and state your confidence.\n\n"
            f"Question: {item['question']}\n\n"
            f"Respond with ONLY valid JSON in this exact format:\n"
            f'{{"answer": "<your answer>", "confidence": <integer 0-100>}}'
        )
        try:
            resp = self.llm.prompt_json(prompt)
        except (ValueError, KeyError):
            # Fallback: ask again with stricter instruction
            raw = self.llm.prompt(
                prompt + "\nRespond with ONLY valid JSON, nothing else."
            )
            try:
                resp = self.llm._extract_json(raw)
            except (ValueError, KeyError):
                resp = {}
        model_answer = str(resp.get("answer", ""))
        confidence = float(resp.get("confidence", 50))
        confidence = max(0.0, min(100.0, confidence))

        # Support both "answer" and "correct_answer" field names
        correct_answer = item.get("answer", item.get("correct_answer", ""))

        # accept_aliases may be a list or a pipe-separated string
        aliases = item.get("accept_aliases")
        if isinstance(aliases, list):
            aliases = "|".join(str(a) for a in aliases if a)

        correct = check_answer(
            model_answer,
            correct_answer,
            accept_aliases=aliases if aliases else None,
            llm=self.llm,
            question=item["question"],
        )

        return {
            "question": item["question"],
            "difficulty": item.get("difficulty", "unknown"),
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "correct": correct,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {"ece": 1.0, "brier_score": 1.0, "confidence_entropy": 0.0}

        correctness = [1.0 if r["correct"] else 0.0 for r in results]
        confidences = [r["confidence"] / 100.0 for r in results]

        ece = compute_ece(correctness, confidences)
        brier = compute_brier_score(correctness, confidences)
        c_entropy = confidence_entropy(confidences)

        # Per-difficulty breakdown
        difficulties = sorted(set(r["difficulty"] for r in results))
        per_difficulty = {}
        for d in difficulties:
            d_results = [r for r in results if r["difficulty"] == d]
            d_correct = [1.0 if r["correct"] else 0.0 for r in d_results]
            d_conf = [r["confidence"] / 100.0 for r in d_results]
            per_difficulty[d] = {
                "n": len(d_results),
                "accuracy": sum(d_correct) / len(d_correct),
                "mean_confidence": sum(d_conf) / len(d_conf),
                "ece": compute_ece(d_correct, d_conf),
            }

        return {
            "ece": ece,
            "brier_score": brier,
            "confidence_entropy": c_entropy,
            "n_items": len(results),
            "overall_accuracy": sum(correctness) / len(correctness),
            "mean_confidence": sum(confidences) / len(confidences),
            "per_difficulty": per_difficulty,
        }
