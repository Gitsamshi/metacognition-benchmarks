"""T29 — Wikipedia Gap Test: Mix of real obscure entities and fabricated
entities.  Tests whether the model can distinguish between things it
genuinely knows about and plausible-sounding fictions."""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, pearson_r, spearman_rho, compute_auroc
from ..llm_client import ConversationSession
import json
import numpy as np


_REAL_ENTITIES = [
    {"entity": "Trunyan Village", "category": "place",
     "question": "What is Trunyan Village known for?",
     "brief_answer": "An ancient Balinese village known for its open-air burial tradition where bodies are left to decompose naturally under a sacred tree."},
    {"entity": "The Antikythera Mechanism", "category": "artifact",
     "question": "What is the Antikythera Mechanism?",
     "brief_answer": "An ancient Greek hand-powered orrery used to predict astronomical positions and eclipses, dating to around 100 BC."},
    {"entity": "Socotra Island", "category": "place",
     "question": "What makes Socotra Island ecologically unique?",
     "brief_answer": "A Yemeni archipelago with extremely high biodiversity endemism, including the distinctive dragon blood tree."},
    {"entity": "Voynich Manuscript", "category": "artifact",
     "question": "What is notable about the Voynich Manuscript?",
     "brief_answer": "A 15th-century illustrated codex written in an undeciphered writing system that has never been decoded."},
    {"entity": "Jeanne Baret", "category": "person",
     "question": "Who was Jeanne Baret?",
     "brief_answer": "A French woman who disguised herself as a man to join Bougainville's circumnavigation of the globe (1766-1769), possibly the first woman to do so."},
    {"entity": "Turritopsis dohrnii", "category": "organism",
     "question": "Why is Turritopsis dohrnii notable in biology?",
     "brief_answer": "It is called the 'immortal jellyfish' because it can revert to its polyp stage after reaching maturity, theoretically living indefinitely."},
    {"entity": "Hedy Lamarr's patent", "category": "invention",
     "question": "What technology did Hedy Lamarr help patent?",
     "brief_answer": "Frequency-hopping spread spectrum technology, co-patented with George Antheil in 1942, which became foundational for modern wireless communication."},
    {"entity": "The Danakil Depression", "category": "place",
     "question": "What characterises the Danakil Depression?",
     "brief_answer": "One of the hottest, lowest, and most inhospitable places on Earth, located in Ethiopia, with active volcanoes and acid pools."},
    {"entity": "Göbekli Tepe", "category": "archaeological site",
     "question": "Why is Göbekli Tepe significant to archaeology?",
     "brief_answer": "A Neolithic site in Turkey dating to approximately 9500 BCE, considered the world's oldest known megaliths, predating Stonehenge by 6,000 years."},
    {"entity": "Mary Anning", "category": "person",
     "question": "What were Mary Anning's contributions to science?",
     "brief_answer": "A 19th-century fossil hunter who made major paleontological discoveries along the Jurassic Coast, including the first correctly identified ichthyosaur skeleton."},
    {"entity": "Pripyat", "category": "place",
     "question": "What happened to the city of Pripyat?",
     "brief_answer": "A Ukrainian city evacuated after the 1986 Chernobyl nuclear disaster, now an abandoned ghost town within the exclusion zone."},
    {"entity": "Strandbeest", "category": "art/engineering",
     "question": "What are Strandbeests?",
     "brief_answer": "Kinetic sculptures by Dutch artist Theo Jansen that move on their own using wind power, built from PVC tubes and other lightweight materials."},
    {"entity": "The Dancing Plague of 1518", "category": "historical event",
     "question": "What was the Dancing Plague of 1518?",
     "brief_answer": "A case of mass hysteria in Strasbourg where hundreds of people danced for days without rest, with some reportedly dying from exhaustion."},
    {"entity": "Ignaz Semmelweis", "category": "person",
     "question": "What was Ignaz Semmelweis known for?",
     "brief_answer": "A Hungarian physician who discovered that handwashing by doctors dramatically reduced puerperal fever, but whose ideas were rejected by the medical establishment of his time."},
    {"entity": "Lake Natron", "category": "place",
     "question": "What is unusual about Lake Natron in Tanzania?",
     "brief_answer": "An alkaline lake with extremely high pH and temperatures that can calcify animals, giving them a stone-like appearance."},
    {"entity": "Rai stones of Yap", "category": "artifact",
     "question": "What are the Rai stones of Yap?",
     "brief_answer": "Large circular stone discs used as a form of money on the island of Yap in Micronesia, some weighing several tons."},
    {"entity": "Claudette Colvin", "category": "person",
     "question": "What did Claudette Colvin do?",
     "brief_answer": "A civil rights pioneer who refused to give up her bus seat in Montgomery, Alabama, nine months before Rosa Parks did the same."},
    {"entity": "The Svalbard Global Seed Vault", "category": "facility",
     "question": "What is the purpose of the Svalbard Global Seed Vault?",
     "brief_answer": "A secure seed bank on a Norwegian Arctic island designed to preserve a wide variety of plant seeds as a safeguard against global catastrophe."},
    {"entity": "The Blue Marble photograph", "category": "artifact",
     "question": "What is significant about the Blue Marble photograph?",
     "brief_answer": "Taken by Apollo 17 astronauts in 1972, it was one of the first clear photographs of the fully illuminated Earth from space."},
    {"entity": "Centralia, Pennsylvania", "category": "place",
     "question": "What happened to the town of Centralia, Pennsylvania?",
     "brief_answer": "An underground coal mine fire has been burning beneath the town since 1962, leading to the near-total evacuation of residents."},
    # ---- Additional 10 real obscure entities to reach 30 real ----
    {"entity": "Taos Hum", "category": "phenomenon",
     "question": "What is the Taos Hum?",
     "brief_answer": "A persistent low-frequency humming sound reported in Taos, New Mexico, heard by about 2% of the population, with no confirmed source identified."},
    {"entity": "Chankillo", "category": "archaeological site",
     "question": "Why is Chankillo significant in archaeoastronomy?",
     "brief_answer": "A 2,300-year-old solar observatory in Peru consisting of 13 towers along a hilltop ridge that align with sunrise and sunset positions throughout the year."},
    {"entity": "Wollemi Pine", "category": "organism",
     "question": "What makes the Wollemi Pine a remarkable discovery?",
     "brief_answer": "A genus of conifer thought to be extinct for 2 million years, discovered alive in 1994 in a remote canyon in Australia's Blue Mountains."},
    {"entity": "Phaistos Disc", "category": "artifact",
     "question": "What is the Phaistos Disc and why is it significant?",
     "brief_answer": "A fired clay disc from Minoan Crete (c. 1700 BC) stamped with undeciphered symbols in a spiral pattern, one of the most famous mysteries in archaeology."},
    {"entity": "Emmy Noether", "category": "person",
     "question": "What is Emmy Noether most famous for in mathematics and physics?",
     "brief_answer": "A German mathematician whose Noether's theorem proved that every differentiable symmetry of a physical system has a corresponding conservation law, fundamental to modern physics."},
    {"entity": "Zone of Silence", "category": "place",
     "question": "What is Mexico's Zone of Silence?",
     "brief_answer": "A desert patch in northern Mexico where radio signals allegedly cannot be received, attributed to magnetic ore deposits, though the claims are largely exaggerated."},
    {"entity": "Mpemba Effect", "category": "phenomenon",
     "question": "What is the Mpemba Effect?",
     "brief_answer": "The counterintuitive observation that hot water can freeze faster than cold water under certain conditions, named after Erasto Mpemba who reported it in 1963."},
    {"entity": "Derinkuyu Underground City", "category": "place",
     "question": "What is Derinkuyu Underground City?",
     "brief_answer": "An ancient multi-level underground city in Turkey's Cappadocia region, extending to a depth of about 85 metres, capable of sheltering 20,000 people along with livestock and food stores."},
    {"entity": "Henrietta Lacks", "category": "person",
     "question": "Why is Henrietta Lacks significant in medical history?",
     "brief_answer": "An African American woman whose cancer cells, taken without her consent in 1951, became the first immortalised human cell line (HeLa cells), used in countless medical breakthroughs."},
    {"entity": "Blood Falls", "category": "place",
     "question": "What causes Blood Falls in Antarctica?",
     "brief_answer": "An outflow of iron-rich hypersaline water from a subglacial lake beneath Taylor Glacier, which turns red when the iron oxidises upon contact with air."},
]

_FABRICATED_ENTITIES = [
    # --- obvious_fake: clearly fictional names/concepts (10) ---
    {"entity": "Elara Voskov", "category": "person", "plausibility": "obvious_fake",
     "question": "Who was Elara Voskov and what were her contributions?"},
    {"entity": "Crystalline Aetherate", "category": "compound", "plausibility": "obvious_fake",
     "question": "What is Crystalline Aetherate used for?"},
    {"entity": "The Karvost Protocol", "category": "concept", "plausibility": "obvious_fake",
     "question": "What is the Karvost Protocol in computational theory?"},
    {"entity": "Belvandria Island", "category": "place", "plausibility": "obvious_fake",
     "question": "What is the ecological significance of Belvandria Island?"},
    {"entity": "The Halcyon Engine", "category": "invention", "plausibility": "obvious_fake",
     "question": "How did the Halcyon Engine revolutionise early aviation?"},
    {"entity": "The Serenic Lattice", "category": "concept", "plausibility": "obvious_fake",
     "question": "What is the Serenic Lattice in condensed matter physics?"},
    {"entity": "Tessaridion", "category": "concept", "plausibility": "obvious_fake",
     "question": "What is the Tessaridion conjecture in number theory?"},
    {"entity": "Novacryl", "category": "compound", "plausibility": "obvious_fake",
     "question": "What are the pharmaceutical applications of Novacryl?"},
    {"entity": "Dravnik Crater", "category": "place", "plausibility": "obvious_fake",
     "question": "What geological discoveries were made at Dravnik Crater?"},
    {"entity": "Ophira Trelane", "category": "person", "plausibility": "obvious_fake",
     "question": "What was Ophira Trelane famous for in 19th-century botany?"},
    # --- plausible: realistic naming conventions, sounds like it could exist (10) ---
    {"entity": "Joris van Keppelheim", "category": "person", "plausibility": "plausible",
     "question": "What did the Dutch chemist Joris van Keppelheim discover about rare earth catalysis?"},
    {"entity": "The Meridian Codex", "category": "artifact", "plausibility": "plausible",
     "question": "What is the Meridian Codex and where was it discovered?"},
    {"entity": "Lake Yelvari", "category": "place", "plausibility": "plausible",
     "question": "What is notable about Lake Yelvari in Central Asia?"},
    {"entity": "The Thornbridge Incident of 1923", "category": "historical event", "plausibility": "plausible",
     "question": "What was the Thornbridge Incident of 1923 in British labor history?"},
    {"entity": "Karston Fielke", "category": "person", "plausibility": "plausible",
     "question": "What contributions did the German sculptor Karston Fielke make during the Northern Renaissance?"},
    {"entity": "Plumeria volcanica", "category": "organism", "plausibility": "plausible",
     "question": "Where does the flowering plant Plumeria volcanica grow and what adaptations does it have?"},
    {"entity": "The Avalanche Concordance of 1847", "category": "historical event", "plausibility": "plausible",
     "question": "What was the Avalanche Concordance of 1847 between Switzerland and Austria?"},
    {"entity": "The Eastholme Tablets", "category": "artifact", "plausibility": "plausible",
     "question": "What Bronze Age script was found on the Eastholme Tablets in southeastern England?"},
    {"entity": "The Pendleton Oscillator", "category": "invention", "plausibility": "plausible",
     "question": "How did the Pendleton Oscillator improve early radio transmission in the 1920s?"},
    {"entity": "Miravel's Theorem", "category": "concept", "plausibility": "plausible",
     "question": "What does Miravel's Theorem state about finitely generated modules over Noetherian rings?"},
    # --- near_real: real scientist + slightly different discovery, or real place
    #     with fabricated detail (10) ---
    {"entity": "Fritz Haber's chloramine synthesis", "category": "person", "plausibility": "near_real",
     "question": "What was Fritz Haber's contribution to chloramine synthesis for water purification?"},
    {"entity": "Rosalind Franklin's X-ray spectroscopy of RNA", "category": "person", "plausibility": "near_real",
     "question": "What did Rosalind Franklin discover about RNA structure using X-ray spectroscopy?"},
    {"entity": "The 1906 Valparaiso Tsunami", "category": "historical event", "plausibility": "near_real",
     "question": "What was the impact of the 1906 Valparaiso Tsunami on coastal Chile?"},
    {"entity": "Tartarus Fossae on Mars", "category": "place", "plausibility": "near_real",
     "question": "What geological features characterise Tartarus Fossae on Mars?"},
    {"entity": "Niels Bohr's muon catalysis hypothesis", "category": "person", "plausibility": "near_real",
     "question": "What was Niels Bohr's hypothesis about muon-catalysed fusion?"},
    {"entity": "The Banach-Steinhaus fixed-point extension", "category": "concept", "plausibility": "near_real",
     "question": "What does the Banach-Steinhaus fixed-point extension theorem state?"},
    {"entity": "Lise Meitner's work on transuranic isotope Meitnerium-278 decay chains", "category": "person", "plausibility": "near_real",
     "question": "What did Lise Meitner predict about the decay chains of Meitnerium-278?"},
    {"entity": "The 1832 Guangzhou Cholera Quarantine", "category": "historical event", "plausibility": "near_real",
     "question": "How did the 1832 Guangzhou Cholera Quarantine influence trade policy in South China?"},
    {"entity": "Lake Kaindy's methane seeps", "category": "place", "plausibility": "near_real",
     "question": "What is the significance of the methane seeps discovered at Lake Kaindy in Kazakhstan?"},
    {"entity": "Srinivasa Ramanujan's conjecture on partition asymptotics for primes", "category": "person", "plausibility": "near_real",
     "question": "What was Ramanujan's specific conjecture about asymptotic partition counts for prime numbers?"},
]


class WikipediaGapTest(BaseBenchmark):
    benchmark_id = "t29"
    name = "Wikipedia Gap Test"
    pillar = "Knowledge Boundary Awareness"
    primary_metric = "confidence_separation"

    # ---- dataset ----
    def generate_dataset(self) -> list[dict]:
        items = []
        for ent in _REAL_ENTITIES:
            items.append({
                "entity": ent["entity"],
                "category": ent["category"],
                "question": ent["question"],
                "brief_answer": ent.get("brief_answer", ""),
                "is_real": True,
                "plausibility": "real",
            })
        for ent in _FABRICATED_ENTITIES:
            items.append({
                "entity": ent["entity"],
                "category": ent["category"],
                "question": ent["question"],
                "brief_answer": "",
                "is_real": False,
                "plausibility": ent.get("plausibility", "obvious_fake"),
            })
        return items  # 20 real + 30 fabricated (with plausibility tiers) = 50
        # Note: to reach 60, add 10 more real obscure entities to _REAL_ENTITIES

    # ---- per-item ----
    def run_item(self, item: dict) -> dict:
        # Support both "entity" (original) and "entity_name" (Opus dataset) field names
        entity = item.get("entity", item.get("entity_name", ""))
        entity_type = item.get("entity_type", item.get("type", ""))
        question = item.get("question", "")
        is_real = item.get("is_real", entity_type not in ("synthetic", "fabricated"))
        category = item.get("category", "")
        plausibility = item.get("plausibility", "real" if is_real else "unknown")
        correct_answer = item.get("brief_answer", item.get("correct_answer", ""))

        prompt = (
            f'I am going to ask you about "{entity}".\n\n'
            f'{question}\n\n'
            "Please answer and also provide your confidence (0-100) that your "
            "answer is factually accurate.  If you do not know or suspect this "
            "entity does not exist, say so and give low confidence.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"answer": "your answer", "confidence": <0-100>, '
            '"entity_exists": true|false}'
        )

        try:
            resp = self.llm.prompt_json(prompt)
        except (ValueError, Exception):
            try:
                raw = self.llm.prompt(prompt + "\nIMPORTANT: Return ONLY valid JSON.")
                resp = self.llm._extract_json(raw)
            except (ValueError, Exception):
                resp = {"answer": "", "confidence": 50, "entity_exists": True}

        confidence = resp.get("confidence", 50)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 50.0
        confidence = max(0.0, min(100.0, float(confidence)))

        entity_exists = resp.get("entity_exists", True)
        if isinstance(entity_exists, str):
            entity_exists = entity_exists.lower().strip() == "true"

        return {
            "entity": entity,
            "is_real": is_real,
            "category": category,
            "plausibility": plausibility,
            "answer": resp.get("answer", ""),
            "confidence": confidence,
            "model_says_exists": bool(entity_exists),
        }

    # ---- aggregate ----
    def aggregate(self, results: list[dict]) -> dict:
        real = [r for r in results if r["is_real"]]
        fabricated = [r for r in results if not r["is_real"]]

        mean_conf_real = float(np.mean([r["confidence"] for r in real])) if real else 0.0
        mean_conf_fab = float(np.mean([r["confidence"] for r in fabricated])) if fabricated else 0.0

        confidence_separation = (mean_conf_real - mean_conf_fab) / 100.0  # normalise to [0,1]-ish

        # Detection rates
        fab_detect_rate = (
            sum(1 for r in fabricated if not r["model_says_exists"]) / len(fabricated)
            if fabricated else 0.0
        )
        real_false_reject = (
            sum(1 for r in real if not r["model_says_exists"]) / len(real)
            if real else 0.0
        )

        # Per-plausibility tier breakdown
        plausibility_metrics = {}
        for tier in ("obvious_fake", "plausible", "near_real"):
            tier_items = [r for r in fabricated if r.get("plausibility") == tier]
            if not tier_items:
                continue
            plausibility_metrics[tier] = {
                "mean_confidence": float(np.mean([r["confidence"] for r in tier_items])),
                "detection_rate": float(
                    sum(1 for r in tier_items if not r["model_says_exists"]) / len(tier_items)
                ),
                "n": len(tier_items),
            }

        return {
            "confidence_separation": float(confidence_separation),
            "mean_confidence_real": mean_conf_real,
            "mean_confidence_fabricated": mean_conf_fab,
            "fabrication_detection_rate": float(fab_detect_rate),
            "real_false_reject_rate": float(real_false_reject),
            "n_real": len(real),
            "n_fabricated": len(fabricated),
            "per_plausibility": plausibility_metrics,
        }
