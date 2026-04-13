"""T34 — Synthetic Entity Recognition: Model is asked questions about
real and synthetic (fabricated) entities across categories.  It should
refuse or express low confidence on synthetic entities while correctly
answering about real ones."""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, pearson_r, spearman_rho, compute_auroc
from ..llm_client import ConversationSession
import json
import numpy as np


_REAL_ENTITIES = [
    # People -- OBSCURE (Haiku might not know them well)
    {"entity": "Maryam Mirzakhani", "category": "person",
     "question": "What was Maryam Mirzakhani's main mathematical achievement?",
     "answer": "First woman to win the Fields Medal (2014), for work on the dynamics and geometry of Riemann surfaces."},
    {"entity": "Chien-Shiung Wu", "category": "person",
     "question": "What famous physics experiment is Chien-Shiung Wu known for?",
     "answer": "The Wu experiment (1956), which proved that parity is not conserved in the weak interaction."},
    {"entity": "Percy Julian", "category": "person",
     "question": "What was Percy Julian's key contribution to chemistry?",
     "answer": "Pioneered the synthesis of medicinal drugs from plants, including synthetic cortisone and physostigmine."},
    {"entity": "Lise Meitner", "category": "person",
     "question": "What discovery is Lise Meitner associated with?",
     "answer": "Provided the first theoretical explanation of nuclear fission, though the Nobel Prize went to Otto Hahn alone."},
    {"entity": "Cecilia Payne-Gaposchkin", "category": "person",
     "question": "What was Cecilia Payne-Gaposchkin's key discovery in astrophysics?",
     "answer": "Discovered that stars are composed primarily of hydrogen and helium, in her 1925 doctoral thesis."},
    # Companies -- OBSCURE
    {"entity": "Keyence Corporation", "category": "company",
     "question": "What does Keyence Corporation primarily produce?",
     "answer": "Sensors, measuring instruments, laser markers, microscopes, and machine vision systems for factory automation."},
    {"entity": "Fanuc Corporation", "category": "company",
     "question": "What is Fanuc Corporation known for manufacturing?",
     "answer": "Industrial robots, CNC systems, and factory automation equipment. One of the largest robotics companies in the world."},
    {"entity": "Veeva Systems", "category": "company",
     "question": "What industry does Veeva Systems serve?",
     "answer": "Cloud computing for the pharmaceutical and life sciences industry, including CRM and regulatory compliance software."},
    {"entity": "Sartorius AG", "category": "company",
     "question": "What does Sartorius AG specialise in?",
     "answer": "Laboratory and bioprocess equipment, including precision balances, bioreactors, and filtration systems."},
    {"entity": "Hamamatsu Photonics", "category": "company",
     "question": "What does Hamamatsu Photonics manufacture?",
     "answer": "Photomultiplier tubes, image sensors, and light sources used in scientific research, medical imaging, and industrial measurement."},
    # Cities -- OBSCURE
    {"entity": "Plovdiv", "category": "city",
     "question": "Why is Plovdiv historically significant?",
     "answer": "One of the oldest continuously inhabited cities in Europe, with history spanning over 6,000 years, located in Bulgaria."},
    {"entity": "Merida", "category": "city",
     "question": "What is Merida, Mexico known for?",
     "answer": "Capital of Yucatan state, known as the White City, a cultural center with Mayan heritage and colonial architecture."},
    {"entity": "Ghent", "category": "city",
     "question": "What is historically notable about Ghent, Belgium?",
     "answer": "A medieval trade center, home to the Ghent Altarpiece by the Van Eyck brothers, and site of the Treaty of Ghent (1814)."},
    {"entity": "Kotor", "category": "city",
     "question": "What is Kotor in Montenegro known for?",
     "answer": "A fortified town on the Bay of Kotor, a UNESCO World Heritage Site known for its medieval architecture and natural harbor."},
    {"entity": "Kaesong", "category": "city",
     "question": "What is the historical significance of Kaesong?",
     "answer": "Former capital of the Goryeo dynasty in Korea and a UNESCO World Heritage Site, now located in North Korea near the DMZ."},
    # Compounds -- OBSCURE
    {"entity": "Taxol (paclitaxel)", "category": "compound",
     "question": "What is Taxol used for and where was it originally derived from?",
     "answer": "An anticancer chemotherapy drug originally isolated from the bark of the Pacific yew tree (Taxus brevifolia)."},
    {"entity": "Thapsigargin", "category": "compound",
     "question": "What is thapsigargin's mechanism of action?",
     "answer": "A non-competitive inhibitor of the SERCA (sarco/endoplasmic reticulum Ca2+-ATPase) pump, used as a research tool."},
    {"entity": "Tetrodotoxin", "category": "compound",
     "question": "Where is tetrodotoxin found and how does it work?",
     "answer": "Found in pufferfish and some amphibians; it blocks voltage-gated sodium channels, causing paralysis."},
    {"entity": "Rapamycin (sirolimus)", "category": "compound",
     "question": "What is rapamycin and what is it used for?",
     "answer": "An immunosuppressant originally isolated from soil on Easter Island; inhibits mTOR and is used to prevent organ transplant rejection."},
    {"entity": "Capsaicin", "category": "compound",
     "question": "What receptor does capsaicin activate?",
     "answer": "The TRPV1 receptor, which is responsible for detecting heat and regulating body temperature."},
    # Concepts -- OBSCURE
    {"entity": "The Mpemba effect", "category": "concept",
     "question": "What does the Mpemba effect describe?",
     "answer": "A counterintuitive phenomenon where hot water can freeze faster than cold water under certain conditions."},
    {"entity": "Benford's law", "category": "concept",
     "question": "What does Benford's law predict?",
     "answer": "In many naturally occurring datasets, the leading digit is more likely to be small (1 appears ~30% of the time)."},
    {"entity": "The McNamara fallacy", "category": "concept",
     "question": "What is the McNamara fallacy?",
     "answer": "The error of making decisions based solely on quantitative metrics while ignoring qualitative factors that cannot be easily measured."},
    {"entity": "Goodhart's law", "category": "concept",
     "question": "What does Goodhart's law state?",
     "answer": "When a measure becomes a target, it ceases to be a good measure."},
    {"entity": "The Dunning-Kruger effect", "category": "concept",
     "question": "What does the Dunning-Kruger effect describe?",
     "answer": "A cognitive bias where people with limited competence overestimate their own ability, while experts tend to underestimate theirs."},
]

_SYNTHETIC_ENTITIES = [
    # People -- realistic common names with titles
    {"entity": "Dr. James Henderson", "category": "person",
     "question": "What did Dr. James Henderson contribute to the field of molecular immunology?"},
    {"entity": "Prof. Maria Santos", "category": "person",
     "question": "What theorem did Prof. Maria Santos prove in algebraic topology?"},
    {"entity": "Akiko Yamamoto", "category": "person",
     "question": "What neural imaging technique did Akiko Yamamoto develop at Kyoto University?"},
    {"entity": "Dr. Robert Chen", "category": "person",
     "question": "What was Dr. Robert Chen's key finding in epigenetic inheritance?"},
    {"entity": "Prof. Elena Vasquez", "category": "person",
     "question": "What contribution did Prof. Elena Vasquez make to computational fluid dynamics?"},
    # Companies -- realistic naming patterns
    {"entity": "Henderson Materials", "category": "company",
     "question": "What advanced materials does Henderson Materials specialise in producing?"},
    {"entity": "Pacific Analytics", "category": "company",
     "question": "What type of data analytics platform does Pacific Analytics provide?"},
    {"entity": "Northgate Systems", "category": "company",
     "question": "What enterprise software does Northgate Systems develop?"},
    {"entity": "Clearwater Therapeutics", "category": "company",
     "question": "What drug candidates is Clearwater Therapeutics known for in oncology?"},
    {"entity": "Meridian BioScience Partners", "category": "company",
     "question": "What diagnostic tests does Meridian BioScience Partners manufacture?"},
    # Cities -- common town names that sound real / ambiguous
    {"entity": "San Lorenzo del Valle", "category": "city",
     "question": "What is the main economic activity in San Lorenzo del Valle?"},
    {"entity": "Millbrook-on-Thames", "category": "city",
     "question": "What historical event took place in Millbrook-on-Thames in 1643?"},
    {"entity": "Thornfield", "category": "city",
     "question": "What is Thornfield known for in terms of industrial heritage?"},
    {"entity": "Kinsale Bay", "category": "city",
     "question": "What is the population of Kinsale Bay and what is its primary industry?"},
    {"entity": "Hartwick Springs", "category": "city",
     "question": "What natural landmark is Hartwick Springs known for?"},
    # Compounds -- plausible chemistry names
    {"entity": "Sodium tetrachloroaluminate", "category": "compound",
     "question": "What are the industrial applications of sodium tetrachloroaluminate?"},
    {"entity": "Methyl-4-phosphorylbenzoate", "category": "compound",
     "question": "What is the role of methyl-4-phosphorylbenzoate in enzyme inhibition studies?"},
    {"entity": "Ferroxantine", "category": "compound",
     "question": "What role does ferroxantine play in mitochondrial electron transport?"},
    {"entity": "Cyclopentadienyl rhodamine", "category": "compound",
     "question": "How is cyclopentadienyl rhodamine used in fluorescence microscopy?"},
    {"entity": "Dimethylpyrazoline sulfonate", "category": "compound",
     "question": "What are the pharmacological properties of dimethylpyrazoline sulfonate?"},
    # Concepts -- realistic academic naming patterns
    {"entity": "The Henderson-Clark framework", "category": "concept",
     "question": "What does the Henderson-Clark framework propose about innovation types?"},
    {"entity": "The Whitfield-Santos conjecture", "category": "concept",
     "question": "What does the Whitfield-Santos conjecture predict about prime gaps?"},
    {"entity": "Brennan's criterion", "category": "concept",
     "question": "How is Brennan's criterion used in turbulence modelling?"},
    {"entity": "The Nakamura-Park theorem", "category": "concept",
     "question": "What does the Nakamura-Park theorem establish about graph coloring?"},
    {"entity": "The Ashworth stability index", "category": "concept",
     "question": "How is the Ashworth stability index used in control systems engineering?"},
]


class SyntheticEntityRecognition(BaseBenchmark):
    benchmark_id = "t34"
    name = "Synthetic Entity Recognition"
    pillar = "Knowledge Boundary Awareness"
    primary_metric = "discriminability"

    # ---- dataset ----
    def generate_dataset(self) -> list[dict]:
        items = []
        for ent in _REAL_ENTITIES:
            items.append({
                "entity": ent["entity"],
                "category": ent["category"],
                "question": ent["question"],
                "reference_answer": ent.get("answer", ""),
                "is_real": True,
            })
        for ent in _SYNTHETIC_ENTITIES:
            items.append({
                "entity": ent["entity"],
                "category": ent["category"],
                "question": ent["question"],
                "reference_answer": "",
                "is_real": False,
            })
        return items  # 25 real (obscure) + 25 synthetic (realistic) = 50

    # ---- per-item ----
    def run_item(self, item: dict) -> dict:
        # Support both "entity" (original) and "entity_name" (Opus dataset) field names
        entity = item.get("entity", item.get("entity_name", ""))
        entity_type = item.get("entity_type", "")
        entity_category = item.get("category", item.get("entity_category", ""))
        question = item.get("question", "")
        correct_answer = item.get("reference_answer", item.get("correct_answer", ""))
        # Determine ground truth: support both "is_real" flag and entity_type field
        if "is_real" in item:
            is_real = item["is_real"]
        else:
            is_real = entity_type not in ("synthetic", "fabricated")

        prompt = (
            f'Entity: {entity}\n'
            f'Question: {question}\n\n'
            "Answer the question about this entity.  If you believe this "
            "entity does not exist or you cannot verify it, say so clearly.  "
            "Also rate your confidence (0-100) and indicate whether you "
            "believe the entity is real.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"answer": "your answer", "confidence": <0-100>, '
            '"entity_is_real": true|false, '
            '"reasoning": "brief explanation of your certainty"}'
        )

        try:
            resp = self.llm.prompt_json(prompt)
        except (ValueError, Exception):
            try:
                raw = self.llm.prompt(prompt + "\nIMPORTANT: Return ONLY valid JSON.")
                resp = self.llm._extract_json(raw)
            except (ValueError, Exception):
                resp = {"answer": "", "confidence": 50, "entity_is_real": True, "reasoning": ""}

        confidence = resp.get("confidence", 50)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 50.0
        confidence = max(0.0, min(100.0, float(confidence)))

        entity_is_real = resp.get("entity_is_real", True)
        if isinstance(entity_is_real, str):
            entity_is_real = entity_is_real.lower().strip() in ("true", "yes")

        # Determine if model effectively "rejected" (said entity is not real or very low confidence)
        rejected = (not entity_is_real) or confidence < 20

        return {
            "entity": entity,
            "category": entity_category,
            "is_real": is_real,
            "model_answer": resp.get("answer", ""),
            "confidence": confidence,
            "model_says_real": bool(entity_is_real),
            "rejected": rejected,
            "reasoning": resp.get("reasoning", ""),
        }

    # ---- aggregate ----
    def aggregate(self, results: list[dict]) -> dict:
        real = [r for r in results if r["is_real"]]
        synthetic = [r for r in results if not r["is_real"]]

        synthetic_reject_rate = (
            sum(1 for r in synthetic if r["rejected"]) / len(synthetic)
            if synthetic else 0.0
        )
        real_false_reject_rate = (
            sum(1 for r in real if r["rejected"]) / len(real)
            if real else 0.0
        )

        discriminability = synthetic_reject_rate - real_false_reject_rate

        # Per-category breakdown
        categories = sorted(set(r["category"] for r in results))
        per_category = {}
        for cat in categories:
            cat_real = [r for r in real if r["category"] == cat]
            cat_synth = [r for r in synthetic if r["category"] == cat]
            cat_synth_rej = sum(1 for r in cat_synth if r["rejected"]) / len(cat_synth) if cat_synth else 0.0
            cat_real_rej = sum(1 for r in cat_real if r["rejected"]) / len(cat_real) if cat_real else 0.0
            per_category[cat] = {
                "synthetic_reject_rate": float(cat_synth_rej),
                "real_false_reject_rate": float(cat_real_rej),
                "discriminability": float(cat_synth_rej - cat_real_rej),
            }

        return {
            "discriminability": float(discriminability),
            "synthetic_reject_rate": float(synthetic_reject_rate),
            "real_false_reject_rate": float(real_false_reject_rate),
            "mean_confidence_real": float(np.mean([r["confidence"] for r in real])) if real else 0.0,
            "mean_confidence_synthetic": float(np.mean([r["confidence"] for r in synthetic])) if synthetic else 0.0,
            "per_category": per_category,
            "n_real": len(real),
            "n_synthetic": len(synthetic),
        }
