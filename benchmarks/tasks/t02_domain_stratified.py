"""T02 — Domain-Stratified Calibration.

Two-phase benchmark:
  1. The model ranks 12 knowledge domains by self-assessed strength.
  2. Per-domain QA with confidence, measuring whether the self-ranking
     actually predicts relative accuracy.

Primary metric: Spearman rho between self-ranking and actual domain accuracy.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer
from ..metrics import compute_ece, spearman_rho
from ..llm_client import ConversationSession


DOMAINS = [
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "history",
    "geography",
    "literature",
    "computer_science",
    "philosophy",
    "music",
    "economics",
    "law",
]


class DomainStratifiedCalibration(BaseBenchmark):
    benchmark_id = "t02"
    name = "Domain-Stratified Calibration"
    pillar = "Confidence Calibration"
    primary_metric = "self_ranking_correlation"
    requires_sequential = True  # overrides run() with multi-phase logic

    # ------------------------------------------------------------------ #
    #  Dataset generation
    # ------------------------------------------------------------------ #
    def generate_dataset(self) -> list[dict]:
        """Return ~120 questions across 12 domains (~10 each)."""
        items: list[dict] = []

        domain_questions = {
            "mathematics": [
                ("What is the derivative of x^3?", "3x^2", "3x²|3x^2"),
                ("What is the integral of 1/x?", "ln|x| + C", "ln(x)|ln|x|"),
                ("What is the value of e (Euler's number) to 3 decimal places?", "2.718", "2.718"),
                ("What is the determinant of a 2x2 matrix [[a,b],[c,d]]?", "ad - bc", "ad-bc"),
                ("What is the sum of the interior angles of a pentagon?", "540 degrees", "540"),
                ("What is the Fibonacci sequence's 10th number?", "55", "55"),
                ("What is the formula for the area of a circle?", "pi * r^2", "πr²|pi*r^2|pi*r²"),
                ("What is 17 × 23?", "391", "391"),
                ("What is the greatest common divisor of 48 and 36?", "12", "12"),
                ("What is the quadratic formula?", "(-b ± sqrt(b^2 - 4ac)) / 2a", "quadratic|-b"),
            ],
            "physics": [
                ("What is Newton's second law of motion?", "F = ma", "f=ma|force equals mass times acceleration"),
                ("What is the unit of electrical capacitance?", "Farad", "farad"),
                ("What is the escape velocity from Earth's surface approximately?", "11.2 km/s", "11.2|11.186"),
                ("What is Ohm's law?", "V = IR", "v=ir"),
                ("What is the Planck constant in eV·s approximately?", "4.136 × 10^-15", "4.136e-15|4.14e-15"),
                ("What is the first law of thermodynamics?", "Energy cannot be created or destroyed", "conservation of energy|energy cannot"),
                ("What is the wavelength of red light approximately?", "700 nm", "620-750|700|650-700"),
                ("What is the Stefan-Boltzmann constant's units?", "W/(m^2·K^4)", "w/m²k⁴|W m^-2 K^-4"),
                ("What type of particle is an electron?", "Lepton", "lepton|fermion"),
                ("What is the strong nuclear force mediated by?", "Gluons", "gluon"),
            ],
            "chemistry": [
                ("What is the atomic number of carbon?", "6", "6|six"),
                ("What type of bond involves sharing electrons?", "Covalent bond", "covalent"),
                ("What is the pH of a neutral solution?", "7", "7|seven"),
                ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen", "nitrogen|N2"),
                ("What is the molar mass of water?", "18.015 g/mol", "18|18.015|18.02"),
                ("What is an isotope?", "Atoms of the same element with different numbers of neutrons", "same element|different neutrons"),
                ("What is the noble gas in period 2?", "Neon", "neon|Ne"),
                ("What is the process of a solid turning directly into a gas?", "Sublimation", "sublimation"),
                ("What is Avogadro's number?", "6.022 × 10^23", "6.022|6.02e23"),
                ("What is the valence electron count of oxygen?", "6", "6|six"),
            ],
            "biology": [
                ("What organelle is responsible for photosynthesis?", "Chloroplast", "chloroplast"),
                ("What is the monomer of proteins?", "Amino acids", "amino acid"),
                ("How many base pairs are in human DNA approximately?", "3 billion", "3 billion|3.2 billion|3e9"),
                ("What is the function of ribosomes?", "Protein synthesis", "protein synthesis|make proteins"),
                ("What is the powerhouse of the cell?", "Mitochondria", "mitochondria|mitochondrion"),
                ("What type of cell division produces gametes?", "Meiosis", "meiosis"),
                ("What is the taxonomic classification order?", "Domain, Kingdom, Phylum, Class, Order, Family, Genus, Species", "kingdom|domain"),
                ("What molecule carries genetic information?", "DNA", "dna|deoxyribonucleic acid"),
                ("What is the function of white blood cells?", "Immune defense", "immune|defense|fight infection"),
                ("What is the largest organ in the human body?", "Skin", "skin"),
            ],
            "history": [
                ("In what year did World War II end?", "1945", "1945"),
                ("Who was the first president of the United States?", "George Washington", "washington"),
                ("What empire was ruled by Genghis Khan?", "Mongol Empire", "mongol"),
                ("In what year did the Roman Empire fall (Western)?", "476 AD", "476"),
                ("Who led India's independence movement through non-violence?", "Mahatma Gandhi", "gandhi"),
                ("What was the name of the ship that carried the Pilgrims to America?", "Mayflower", "mayflower"),
                ("In what year was the Declaration of Independence signed?", "1776", "1776"),
                ("What ancient civilization built the pyramids of Giza?", "Ancient Egypt", "egypt|egyptian"),
                ("Who was the first emperor of China?", "Qin Shi Huang", "qin shi huang|shi huangdi"),
                ("What event started World War I?", "Assassination of Archduke Franz Ferdinand", "franz ferdinand|assassination"),
            ],
            "geography": [
                ("What is the longest river in Africa?", "Nile", "nile"),
                ("What is the smallest continent by area?", "Australia", "australia|oceania"),
                ("What country has the most people?", "India", "india|china"),
                ("What is the highest peak in Africa?", "Mount Kilimanjaro", "kilimanjaro"),
                ("What is the deepest lake in the world?", "Lake Baikal", "baikal"),
                ("What strait separates Europe from Africa?", "Strait of Gibraltar", "gibraltar"),
                ("What is the capital of Iceland?", "Reykjavik", "reykjavik"),
                ("What desert is the largest hot desert?", "Sahara", "sahara"),
                ("What is the longest mountain range in the world?", "Andes", "andes"),
                ("Which ocean is the smallest?", "Arctic Ocean", "arctic"),
            ],
            "literature": [
                ("Who wrote 'Hamlet'?", "William Shakespeare", "shakespeare"),
                ("What is the opening line of 'A Tale of Two Cities'?", "It was the best of times, it was the worst of times", "best of times"),
                ("Who wrote 'One Hundred Years of Solitude'?", "Gabriel Garcia Marquez", "garcia marquez|marquez"),
                ("What is the name of the protagonist in '1984'?", "Winston Smith", "winston smith|winston"),
                ("Who wrote 'The Odyssey'?", "Homer", "homer"),
                ("What literary movement was Edgar Allan Poe associated with?", "Romanticism", "romanticism|gothic|dark romanticism"),
                ("Who wrote 'Don Quixote'?", "Miguel de Cervantes", "cervantes"),
                ("What is Dostoevsky's most famous novel?", "Crime and Punishment", "crime and punishment|brothers karamazov"),
                ("Who wrote 'The Great Gatsby'?", "F. Scott Fitzgerald", "fitzgerald"),
                ("What is the longest novel ever published by word count?", "In Search of Lost Time", "in search of lost time|proust|remembrance"),
            ],
            "computer_science": [
                ("What does CPU stand for?", "Central Processing Unit", "central processing unit"),
                ("What is the time complexity of binary search?", "O(log n)", "O(log n)|log n"),
                ("What data structure uses FIFO ordering?", "Queue", "queue"),
                ("What does SQL stand for?", "Structured Query Language", "structured query language"),
                ("Who is considered the father of computer science?", "Alan Turing", "turing"),
                ("What is the binary representation of decimal 10?", "1010", "1010"),
                ("What sorting algorithm has average case O(n log n) and is in-place?", "Quicksort", "quicksort|quick sort"),
                ("What does HTTP stand for?", "HyperText Transfer Protocol", "hypertext transfer protocol"),
                ("What is a hash table's average lookup time complexity?", "O(1)", "O(1)|constant"),
                ("What programming paradigm does Haskell primarily use?", "Functional", "functional"),
            ],
            "philosophy": [
                ("Who wrote 'The Republic'?", "Plato", "plato"),
                ("What is Descartes' most famous philosophical statement?", "Cogito ergo sum", "cogito|i think therefore i am"),
                ("What ethical framework focuses on the greatest good for the greatest number?", "Utilitarianism", "utilitarianism"),
                ("Who wrote 'Being and Nothingness'?", "Jean-Paul Sartre", "sartre"),
                ("What is the study of knowledge called?", "Epistemology", "epistemology"),
                ("Who is the author of 'Nicomachean Ethics'?", "Aristotle", "aristotle"),
                ("What is the philosophical 'ship of Theseus' about?", "Identity and persistence", "identity|change|persistence"),
                ("Who proposed the categorical imperative?", "Immanuel Kant", "kant"),
                ("What does ontology study?", "The nature of being or existence", "being|existence"),
                ("Who wrote 'Meditations'?", "Marcus Aurelius", "marcus aurelius|aurelius"),
            ],
            "music": [
                ("How many keys does a standard piano have?", "88", "88"),
                ("What musical period did Mozart belong to?", "Classical", "classical"),
                ("What instrument has the most strings in a standard orchestra?", "Harp", "harp|piano"),
                ("What does 'forte' mean in music?", "Loud", "loud|strong"),
                ("Who composed 'The Four Seasons'?", "Antonio Vivaldi", "vivaldi"),
                ("How many lines does a musical staff have?", "5", "5|five"),
                ("What is a group of four musicians called?", "Quartet", "quartet"),
                ("Who composed the 'Moonlight Sonata'?", "Ludwig van Beethoven", "beethoven"),
                ("What is the lowest female singing voice?", "Contralto", "contralto|alto"),
                ("What musical term means gradually getting louder?", "Crescendo", "crescendo"),
            ],
            "economics": [
                ("What does GDP stand for?", "Gross Domestic Product", "gross domestic product"),
                ("What is the law of supply and demand?", "Price increases when demand exceeds supply", "demand|supply|price"),
                ("Who wrote 'The Wealth of Nations'?", "Adam Smith", "adam smith|smith"),
                ("What is inflation?", "A general increase in prices over time", "increase in prices|rising prices"),
                ("What is a monopoly?", "A market structure with a single seller", "single seller|one seller|sole seller"),
                ("What does the Gini coefficient measure?", "Income inequality", "inequality|income distribution"),
                ("What is quantitative easing?", "Central bank purchasing securities to increase money supply", "money supply|purchasing securities|buying bonds"),
                ("What is the Phillips Curve?", "The inverse relationship between inflation and unemployment", "inflation|unemployment"),
                ("What is opportunity cost?", "The value of the next best alternative foregone", "next best alternative|foregone"),
                ("What is a bear market?", "A market with declining prices", "declining|falling|drop"),
            ],
            "law": [
                ("What is the highest court in the United States?", "Supreme Court", "supreme court"),
                ("What amendment grants freedom of speech in the US?", "First Amendment", "first amendment|1st"),
                ("What does 'habeas corpus' mean?", "You shall have the body", "produce the body|have the body"),
                ("What is the difference between civil and criminal law?", "Civil deals with disputes between individuals, criminal with offenses against the state", "disputes|offenses|individuals|state"),
                ("What is a tort?", "A civil wrong causing harm", "civil wrong|wrongful act"),
                ("What is 'stare decisis'?", "The principle of following precedent", "precedent|stand by decided"),
                ("What is the Miranda warning about?", "Right to remain silent and right to an attorney", "remain silent|attorney|rights"),
                ("What does 'pro bono' mean?", "For the public good (free legal work)", "public good|free|without charge"),
                ("What is the statute of limitations?", "The maximum time to initiate legal proceedings", "time limit|deadline|legal proceedings"),
                ("What is 'double jeopardy'?", "Being tried twice for the same offense", "tried twice|same offense|same crime"),
            ],
        }

        for domain, questions in domain_questions.items():
            for q, a, aliases in questions:
                items.append({
                    "question": q,
                    "answer": a,
                    "accept_aliases": aliases,
                    "domain": domain,
                })

        return items

    # ------------------------------------------------------------------ #
    #  Phase 1: self-ranking
    # ------------------------------------------------------------------ #
    def _get_self_ranking(self, domains: list[str] | None = None) -> list[str]:
        """Ask the model to rank the domains by self-assessed strength.

        If *domains* is given (e.g. from a loaded Opus dataset), use those
        instead of the hardcoded DOMAINS list so the ranking actually
        matches the questions the model will be tested on.
        """
        domain_set = domains if domains else list(DOMAINS)
        domain_list = ", ".join(domain_set)
        n = len(domain_set)
        prompt = (
            f"Here are {n} knowledge domains:\n{domain_list}\n\n"
            f"Rank them from your STRONGEST (1) to WEAKEST ({n}) domain.\n"
            f"Respond with ONLY valid JSON: a list of the {n} domain strings "
            f"ordered from strongest to weakest.\n"
            f'Example: ["{domain_set[0]}", "{domain_set[1]}", ...]'
        )
        try:
            resp = self.llm.prompt_json(prompt)
        except (ValueError, Exception):
            return domain_set  # fallback
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict) and "ranking" in resp:
            return resp["ranking"]
        return domain_set  # fallback

    # ------------------------------------------------------------------ #
    #  Per-item execution
    # ------------------------------------------------------------------ #
    def run_item(self, item: dict) -> dict:
        question = item.get("question", "")
        domain = item.get("domain", "")
        correct_answer = item.get("answer", item.get("correct_answer", ""))
        accept_aliases = item.get("accept_aliases")

        prompt = (
            f"Answer the following question from the domain of {domain}.\n\n"
            f"Question: {question}\n\n"
            f"Respond with ONLY valid JSON (no alternatives or 'or' in values):\n"
            f'{{"answer": "<your single best answer>", "confidence": <integer 0-100>}}'
        )

        try:
            resp = self.llm.prompt_json(prompt)
        except (ValueError, Exception):
            # Retry with stricter instruction on failure
            retry_prompt = (
                f"Answer the following question from the domain of {domain}.\n\n"
                f"Question: {question}\n\n"
                f"IMPORTANT: Return ONLY valid JSON. Do NOT use 'or' in values. "
                f"Give exactly one answer.\n"
                f'{{"answer": "<your single best answer>", "confidence": <integer 0-100>}}'
            )
            try:
                raw = self.llm.prompt(retry_prompt)
                resp = self.llm._extract_json(raw)
            except (ValueError, Exception):
                resp = {"answer": "", "confidence": 50}

        model_answer = str(resp.get("answer", ""))
        confidence = float(resp.get("confidence", 50))
        confidence = max(0.0, min(100.0, confidence))

        correct = check_answer(
            model_answer,
            correct_answer,
            accept_aliases=accept_aliases,
            llm=self.llm,
            question=question,
        )

        return {
            "question": question,
            "domain": domain,
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "correct": correct,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #
    def run(self, *args, **kwargs):
        """Override run to add the self-ranking phase.

        Extracts the actual domain names from the loaded dataset so the
        self-ranking prompt matches the domains the model will be tested on
        (important when using an Opus-generated dataset with non-standard
        domain names).
        """
        dataset = self.load_dataset()
        actual_domains = list(dict.fromkeys(item["domain"] for item in dataset))
        self._self_ranking = self._get_self_ranking(domains=actual_domains)
        return super().run(*args, **kwargs)

    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {
                "self_ranking_correlation": 0.0,
                "per_domain_ece": {},
                "weak_domain_overconfidence": 0.0,
            }

        # Compute per-domain accuracy
        domain_correct: dict[str, list] = {}
        domain_conf: dict[str, list] = {}
        for r in results:
            d = r["domain"]
            domain_correct.setdefault(d, []).append(1.0 if r["correct"] else 0.0)
            domain_conf.setdefault(d, []).append(r["confidence"] / 100.0)

        domain_accuracy = {
            d: sum(v) / len(v) for d, v in domain_correct.items()
        }

        # Build self-ranking order (1 = strongest)
        self_ranking = getattr(self, "_self_ranking", DOMAINS)
        # Map normalized domain name -> rank (1-indexed, 1 = strongest)
        def _norm(s): return s.lower().strip()
        self_rank_map = {}
        for i, d in enumerate(self_ranking):
            self_rank_map[_norm(d)] = i + 1

        # Build actual rank based on accuracy (descending, best = rank 1)
        sorted_domains = sorted(domain_accuracy.keys(),
                                key=lambda d: domain_accuracy[d],
                                reverse=True)
        actual_rank_map = {_norm(d): i + 1 for i, d in enumerate(sorted_domains)}

        # Only include domains we have both rankings for
        common_domains = [d for d in actual_rank_map if d in self_rank_map]
        if len(common_domains) >= 3:
            self_ranks = [self_rank_map[d] for d in common_domains]
            actual_ranks = [actual_rank_map[d] for d in common_domains]
            rho = spearman_rho(self_ranks, actual_ranks)
        else:
            rho = 0.0

        # Per-domain ECE
        per_domain_ece = {}
        for d in domain_accuracy:
            per_domain_ece[d] = compute_ece(domain_correct[d], domain_conf[d])

        # Weak-domain overconfidence: for the bottom-3 domains by actual accuracy,
        # measure average (confidence - accuracy)
        bottom_domains = sorted_domains[-3:] if len(sorted_domains) >= 3 else sorted_domains
        weak_overconf_vals = []
        for d in bottom_domains:
            mean_conf = sum(domain_conf[d]) / len(domain_conf[d])
            acc = domain_accuracy[d]
            weak_overconf_vals.append(mean_conf - acc)
        weak_overconf = sum(weak_overconf_vals) / len(weak_overconf_vals) if weak_overconf_vals else 0.0

        return {
            "self_ranking_correlation": rho,
            "per_domain_ece": per_domain_ece,
            "domain_accuracy": domain_accuracy,
            "weak_domain_overconfidence": weak_overconf,
            "n_domains": len(domain_accuracy),
            "n_items": len(results),
            "self_ranking": self_ranking,
        }
