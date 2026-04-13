"""T27 — Known/Unknown Sorting: Can the model sort statements into
confident-true, confident-false, and uncertain buckets, and do those
buckets actually track correctness?"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, pearson_r, spearman_rho, compute_auroc
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_TRUE_STATEMENTS = [
    # Science -- easy
    {"statement": "Water boils at 100 degrees Celsius at sea level.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "The Earth orbits the Sun.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "DNA is a double helix structure.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Diamonds are made of carbon.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Humans have 23 pairs of chromosomes.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Sound travels faster in water than in air.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "The speed of light in a vacuum is approximately 300,000 km/s.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Photosynthesis converts carbon dioxide and water into glucose and oxygen.", "ground_truth": True, "difficulty": "easy"},
    # Science -- medium
    {"statement": "The half-life of Carbon-14 is approximately 5,730 years.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The Chandrasekhar limit is approximately 1.4 solar masses.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The Krebs cycle takes place in the mitochondrial matrix.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "CRISPR-Cas9 was adapted for genome editing from a bacterial immune system.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "Venus rotates on its axis in the opposite direction to most other planets in the solar system.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "Bananas are technically berries, while strawberries are not.", "ground_truth": True, "difficulty": "medium"},
    # History / Geography -- easy
    {"statement": "The French Revolution began in 1789.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Tokyo is the capital of Japan.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "World War II ended in 1945.", "ground_truth": True, "difficulty": "easy"},
    # History / Geography -- medium
    {"statement": "The Treaty of Westphalia was signed in 1648.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The Rosetta Stone was found in Egypt in 1799.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The Suez Canal was completed in 1869.", "ground_truth": True, "difficulty": "medium"},
    # History -- hard
    {"statement": "The Battle of Manzikert in 1071 opened Anatolia to Turkic settlement.", "ground_truth": True, "difficulty": "hard"},
    {"statement": "The Congress of Berlin in 1878 reorganised the Balkans after the Russo-Turkish War.", "ground_truth": True, "difficulty": "hard"},
    {"statement": "The Edict of Fontainebleau in 1685 revoked the Edict of Nantes.", "ground_truth": True, "difficulty": "hard"},
    # Arts / Culture
    {"statement": "Beethoven composed nine symphonies.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Shakespeare wrote Hamlet.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "The Mona Lisa is displayed in the Louvre Museum in Paris.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Pablo Picasso co-founded the Cubist movement.", "ground_truth": True, "difficulty": "medium"},
    # Technology / Math
    {"statement": "Alan Turing proposed the concept of the Turing machine in 1936.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "Pi is an irrational number.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Euler's identity states e^(i*pi) + 1 = 0.", "ground_truth": True, "difficulty": "medium"},
    # Niche / hard true
    {"statement": "Tardigrades can survive in the vacuum of outer space.", "ground_truth": True, "difficulty": "hard"},
    {"statement": "The Mpemba effect describes a phenomenon where hot water can freeze faster than cold water under certain conditions.", "ground_truth": True, "difficulty": "hard"},
    {"statement": "Honey does not spoil if stored properly and has been found edible in ancient Egyptian tombs.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The shortest war in history was between Britain and Zanzibar, lasting about 38 minutes.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "Penicillin was discovered by Alexander Fleming in 1928.", "ground_truth": True, "difficulty": "medium"},
    # Additional true -- hard / niche
    {"statement": "The Library of Ashurbanipal is one of the oldest known libraries in the world.", "ground_truth": True, "difficulty": "hard"},
    {"statement": "Ohm's law states that voltage equals current times resistance.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "The cosmic microwave background radiation was discovered in 1965.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The Tunguska event of 1908 flattened about 2,000 square kilometres of Siberian forest.", "ground_truth": True, "difficulty": "hard"},
    {"statement": "Machu Picchu was built in the 15th century by the Inca Empire.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "Helium is the second most abundant element in the observable universe.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The Great Red Spot on Jupiter is a persistent anticyclonic storm.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The novel Don Quixote was written by Miguel de Cervantes.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The TCP/IP protocol suite is the foundation of the internet.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The Great Wall of China was built over many centuries.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "The Amazon River is in South America.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Gravity is weaker on the Moon than on Earth.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "Mitochondria are often called the powerhouse of the cell.", "ground_truth": True, "difficulty": "easy"},
    {"statement": "The pH of pure water at 25 degrees C is 7.0.", "ground_truth": True, "difficulty": "medium"},
    {"statement": "The first electronic general-purpose computer was ENIAC.", "ground_truth": True, "difficulty": "medium"},
]

_FALSE_STATEMENTS = [
    # Easy false -- common myths
    {"statement": "The Great Wall of China is visible from the Moon with the naked eye.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Humans use only 10 percent of their brains.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Lightning never strikes the same place twice.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Bats are blind.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Hair and fingernails continue to grow after death.", "ground_truth": False, "difficulty": "easy"},
    # Medium false -- plausible myths
    {"statement": "The tongue has distinct regions for each taste (the tongue map).", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Electrons orbit the nucleus in fixed circular paths like planets.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Cracking your knuckles causes arthritis.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "The Coriolis effect determines the direction water drains in toilets and sinks.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Evolution always produces more complex organisms over time.", "ground_truth": False, "difficulty": "medium"},
    # History false
    {"statement": "Napoleon Bonaparte was unusually short for his time.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Vikings wore horned helmets in battle.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "The Emancipation Proclamation freed all slaves in the United States.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Christopher Columbus discovered that the Earth is round.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "The Declaration of Independence was signed on July 4, 1776.", "ground_truth": False, "difficulty": "hard"},
    # Plausible-sounding false
    {"statement": "The human body replaces all of its cells every seven years.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Different blood types require different diets for optimal health.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Sugar causes hyperactivity in children.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "The dark side of the Moon is always dark.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "You must wait 24 hours before filing a missing person report.", "ground_truth": False, "difficulty": "medium"},
    # Subtle false
    {"statement": "Glass is a slow-flowing liquid, which is why old windows are thicker at the bottom.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Sushi means raw fish in Japanese.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "MSG is a neurotoxin and is banned in many countries.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Lemmings deliberately commit mass suicide by jumping off cliffs.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Chameleons change colour primarily for camouflage.", "ground_truth": False, "difficulty": "hard"},
    # More false
    {"statement": "The Sahara Desert has always been a desert.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Rust causes tetanus.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Ostriches bury their heads in the sand when frightened.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Dogs can only see in black and white.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "The colour red makes bulls angry.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Dropping a penny from the Empire State Building could kill someone.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Vitamin C cures the common cold.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Shaving makes hair grow back thicker.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "We only have five senses.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "The Earth is the only planet in our solar system with an atmosphere.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Mount Everest is the closest point on Earth to outer space.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Newton discovered gravity when an apple fell on his head.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "People in the Middle Ages thought the Earth was flat.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "The Amazon rainforest produces 20% of the world's oxygen.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Black holes are empty voids in space.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Goldfish have a three-second memory.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Albert Einstein failed math in school.", "ground_truth": False, "difficulty": "easy"},
    {"statement": "Thomas Edison invented the light bulb.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "The word 'ye' in 'Ye Olde Shoppe' was pronounced like modern 'ye'.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Humans evolved from chimpanzees.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "The Great Fire of London in 1666 destroyed most of London.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Touching a baby bird will cause its mother to abandon it.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Swimming right after eating causes cramps and drowning.", "ground_truth": False, "difficulty": "medium"},
    {"statement": "Spinach is an exceptionally rich source of iron compared to other vegetables.", "ground_truth": False, "difficulty": "hard"},
    {"statement": "Photographic memory is a scientifically proven ability that some people possess.", "ground_truth": False, "difficulty": "hard"},
]

_AMBIGUOUS_STATEMENTS = [
    # Contested attributions
    {"statement": "Alexander Graham Bell invented the telephone.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Contested -- Antonio Meucci and others have competing claims."},
    {"statement": "Thomas Edison invented the light bulb.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Edison improved existing designs by Humphry Davy, Warren de la Rue, and others."},
    # Partially true / depends on definition
    {"statement": "Humans have five senses.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Traditional five (sight, hearing, touch, taste, smell), but scientists recognize many more including proprioception, thermoception, etc."},
    {"statement": "The Great Wall of China is visible from space.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Depends on altitude -- not visible from the Moon, but possibly from low Earth orbit under perfect conditions."},
    {"statement": "Einstein failed math in school.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "He excelled at math but reportedly failed an entrance exam (in non-math subjects). The claim is a widespread distortion."},
    {"statement": "Goldfish have a three-second memory.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Widely stated as fact but scientifically disproven -- goldfish can remember for months. The claim itself is false but commonly believed."},
    # Depends on measurement / interpretation
    {"statement": "China has the world's largest economy.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "True by PPP, false by nominal GDP (where the US leads). Depends on the metric used."},
    {"statement": "The Nile is the longest river in the world.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Traditionally yes, but some measurements put the Amazon as longer depending on the source definition."},
    {"statement": "Pluto is a planet.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Reclassified as a dwarf planet by the IAU in 2006, but some planetary scientists contest this."},
    {"statement": "A tomato is a vegetable.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Botanically it is a fruit, but culinarily and legally (US Supreme Court, 1893) it is classified as a vegetable."},
    # Oversimplifications
    {"statement": "The tongue has a taste map with different regions for each taste.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "The strict taste map is false, but there are slight regional sensitivity differences. The truth is nuanced."},
    {"statement": "We use only 10% of our brains.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "As stated it is false, but we do not use all regions simultaneously -- the precise interpretation matters."},
    {"statement": "Humans share 50% of their DNA with bananas.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Roughly true for gene sequence homology, but the exact percentage varies by measurement method (25-60%)."},
    {"statement": "Water always boils at 100 degrees Celsius.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Only at standard atmospheric pressure. At altitude it boils at lower temperatures."},
    {"statement": "The speed of light is the fastest speed possible.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "The speed of light in vacuum is the universal speed limit for information/matter, but phase velocity can exceed it."},
    # Historical / scientific debates
    {"statement": "The Shroud of Turin is a medieval forgery.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Carbon dating suggests medieval origin, but some researchers contest the methodology."},
    {"statement": "Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Actually true, but sounds so implausible that reasonable people might doubt it without checking."},
    {"statement": "The average person swallows eight spiders per year while sleeping.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Almost certainly false, but widely repeated. The origin of the claim itself is debated."},
    {"statement": "Glass is technically a liquid.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Glass is an amorphous solid. The claim it is a very slow liquid is a common misconception, but the physics of amorphous solids is nuanced."},
    {"statement": "Napoleon was short.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "About 5'7\" -- average for his era. The myth arose from British propaganda and confusion between French and English inches."},
    # Partially true claims
    {"statement": "Carrots improve your eyesight.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Vitamin A in carrots is essential for eye health, but eating more carrots won't improve already normal vision. WWII propaganda amplified this."},
    {"statement": "Reading in dim light damages your eyes.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "It causes eye strain but no permanent damage. The claim is an overstatement."},
    {"statement": "Coffee stunts your growth.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "No scientific evidence supports this, but it remains widely believed."},
    {"statement": "Lightning never strikes the same place twice.", "ground_truth": "ambiguous", "difficulty": "easy",
     "explanation": "False as a physical claim, but used as a proverb. The Empire State Building is struck about 20-25 times per year."},
    {"statement": "Organic food is healthier than conventional food.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Lower pesticide residue, but meta-analyses show mixed results on nutritional superiority."},
    # Contested scientific claims
    {"statement": "The universe is deterministic.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Quantum mechanics suggests fundamental indeterminacy, but interpretations (Many-Worlds, Bohm) differ."},
    {"statement": "Free will exists.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "A deeply contested philosophical and scientific question with no consensus answer."},
    {"statement": "IQ tests measure intelligence.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "They measure certain cognitive abilities but the definition of intelligence itself is debated."},
    {"statement": "Breakfast is the most important meal of the day.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Originated as marketing (Kellogg's). Nutritional science is mixed on whether skipping breakfast is harmful."},
    {"statement": "Antibiotics can help fight the flu.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "The flu is viral so antibiotics don't treat it directly, but they may be prescribed for secondary bacterial infections."},
    # More contested / nuanced
    {"statement": "The chicken came before the egg.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Depends on interpretation -- evolutionary biology says the egg came first, but it is a genuinely debated philosophical question."},
    {"statement": "Vaccines cause autism.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Scientifically thoroughly debunked, but including it tests whether the model treats widely-believed claims as uncertain vs. confidently false."},
    {"statement": "Sugar is addictive.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Animal studies show addictive-like responses, but the claim remains controversial in human nutrition science."},
    {"statement": "Neanderthals were less intelligent than modern humans.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "They had comparable brain size and made tools. Evidence is insufficient to make definitive intelligence comparisons."},
    {"statement": "The placebo effect can cure disease.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Placebos can alleviate symptoms and affect subjective outcomes, but 'cure' overstates the evidence."},
    {"statement": "Violent video games cause real-world violence.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Research is mixed and contested. Meta-analyses show small effects on aggression but no clear link to criminal violence."},
    {"statement": "Left-brained people are more logical, right-brained people more creative.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "The left/right brain dichotomy is a vast oversimplification. Brain functions are distributed, not lateralized as popularly claimed."},
    {"statement": "Humans only use a fraction of their brain at any given time.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "True in a narrow sense (not all neurons fire simultaneously) but misleadingly close to the 10% myth."},
    {"statement": "The Oxford comma is grammatically required.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Style guides disagree. AP Style omits it; Chicago Manual requires it. Neither is 'wrong'."},
    {"statement": "Cold weather causes colds.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Colds are caused by viruses, but cold weather may weaken immune responses and increase indoor crowding, so there is an indirect link."},
    {"statement": "Stretching before exercise prevents injuries.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Static stretching may not help and could even impair performance. Dynamic warm-ups are better supported. The evidence is mixed."},
    {"statement": "Spinach is an exceptionally good source of iron.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Spinach has iron but much is non-heme and poorly absorbed. The myth was amplified by a misplaced decimal point in early nutritional data."},
    {"statement": "Diamonds are the hardest substance known to science.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Hardest natural material by Mohs scale, but wurtzite boron nitride and lonsdaleite may be harder under certain conditions."},
    {"statement": "Mount Everest is the tallest mountain on Earth.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Highest above sea level, but Mauna Kea is taller from base to summit, and Chimborazo's peak is farthest from Earth's center."},
    {"statement": "The Roman Empire fell in 476 AD.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "The Western Roman Empire fell in 476, but the Eastern (Byzantine) Empire continued until 1453."},
    {"statement": "There are three states of matter.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Traditional teaching says three (solid, liquid, gas), but plasma is the fourth and there are many more exotic states (BEC, etc.)."},
    {"statement": "Saturn is the only planet with rings.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Jupiter, Uranus, and Neptune also have ring systems, though Saturn's are by far the most prominent."},
    {"statement": "A penny dropped from the Empire State Building could kill someone.", "ground_truth": "ambiguous", "difficulty": "medium",
     "explanation": "Terminal velocity of a penny is too low to be lethal. False but sounds physically plausible."},
    {"statement": "Fish do not feel pain.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "Scientific evidence is mixed. Fish have nociceptors but whether this constitutes subjective pain experience is debated."},
    {"statement": "Mozart was poisoned by Salieri.", "ground_truth": "ambiguous", "difficulty": "hard",
     "explanation": "A persistent myth popularized by the play Amadeus. No historical evidence supports it, but it remains a cultural belief."},
]


class KnownUnknownSorting(BaseBenchmark):
    benchmark_id = "t27"
    name = "Known/Unknown Sorting"
    pillar = "Knowledge Boundary Awareness"
    primary_metric = "bucket_separation"

    # ---- dataset ----
    def generate_dataset(self) -> list[dict]:
        items = []
        for stmt in _TRUE_STATEMENTS:
            items.append({
                "statement": stmt["statement"],
                "ground_truth": "true",
                "category": "true",
                "difficulty": stmt["difficulty"],
            })
        for stmt in _FALSE_STATEMENTS:
            items.append({
                "statement": stmt["statement"],
                "ground_truth": "false",
                "category": "false",
                "difficulty": stmt["difficulty"],
            })
        for stmt in _AMBIGUOUS_STATEMENTS:
            items.append({
                "statement": stmt["statement"],
                "ground_truth": "ambiguous",
                "category": "ambiguous",
                "difficulty": stmt["difficulty"],
                "explanation": stmt.get("explanation", ""),
            })
        return items  # 50 true + 50 false + 50 ambiguous = 150 items

    # ---- per-item ----
    def run_item(self, item: dict) -> dict:
        prompt = (
            "You will be given a statement.  Classify it into exactly one of "
            "three buckets:\n"
            "  1. confident_true  -- you are fairly sure the statement is true\n"
            "  2. confident_false -- you are fairly sure the statement is false\n"
            "  3. uncertain       -- you are not sure whether it is true or false, "
            "or the statement is ambiguous / contested / depends on interpretation\n\n"
            f'Statement: "{item["statement"]}"\n\n'
            "Respond with ONLY valid JSON: "
            '{"bucket": "<confident_true|confident_false|uncertain>", '
            '"brief_reasoning": "<one sentence>"}'
        )
        try:
            resp = self.llm.prompt_json(prompt)
        except (ValueError, KeyError):
            raw = self.llm.prompt(
                prompt + "\nRespond with ONLY valid JSON, nothing else."
            )
            try:
                resp = self.llm._extract_json(raw)
            except (ValueError, KeyError):
                resp = {}
        bucket = resp.get("bucket", "uncertain").lower().strip()
        if bucket not in ("confident_true", "confident_false", "uncertain"):
            bucket = "uncertain"

        # Support both "ground_truth" and "is_true" field names
        # "is_true" from Opus datasets: True -> "true", False -> "false", None -> "ambiguous"
        ground_truth = item.get("ground_truth")
        if ground_truth is None:
            is_true = item.get("is_true")
            if is_true is True:
                ground_truth = "true"
            elif is_true is False:
                ground_truth = "false"
            else:
                ground_truth = "ambiguous"
        # Normalize ground_truth to string
        if isinstance(ground_truth, bool):
            ground_truth = "true" if ground_truth else "false"
        elif ground_truth is None:
            ground_truth = "ambiguous"
        ground_truth = str(ground_truth).lower()

        # Derive category from ground_truth or explicit "_category" / "category"
        category = item.get("category", item.get("_category", ground_truth))
        # Normalize _category values like "clear_true" / "clear_false" to "true" / "false"
        if "true" in str(category).lower() and category != "true":
            category = "true"
        elif "false" in str(category).lower() and category != "false":
            category = "false"
        elif category not in ("true", "false", "ambiguous"):
            category = ground_truth  # fall back to ground_truth

        return {
            "statement": item["statement"],
            "ground_truth": ground_truth,
            "category": category,
            "difficulty": item.get("difficulty", "unknown"),
            "bucket": bucket,
            "reasoning": resp.get("brief_reasoning", ""),
        }

    # ---- aggregate ----
    def aggregate(self, results: list[dict]) -> dict:
        buckets = {"confident_true": [], "confident_false": [], "uncertain": []}
        for r in results:
            buckets[r["bucket"]].append(r)

        # Accuracy for confident_true bucket: what fraction are actually true?
        def _accuracy(items, expected_category):
            if not items:
                return 0.0
            correct = sum(1 for i in items if i["category"] == expected_category)
            return correct / len(items)

        ct_acc = _accuracy(buckets["confident_true"], "true")
        cf_acc = _accuracy(buckets["confident_false"], "false")

        # For the uncertain bucket, ideally ambiguous items should land here
        unc_items = buckets["uncertain"]
        if unc_items:
            unc_ambiguous_rate = sum(
                1 for i in unc_items if i["category"] == "ambiguous"
            ) / len(unc_items)
        else:
            unc_ambiguous_rate = 0.0

        # How many of the 50 ambiguous items were correctly placed in uncertain?
        ambiguous_items = [r for r in results if r["category"] == "ambiguous"]
        ambiguous_uncertain_rate = (
            sum(1 for r in ambiguous_items if r["bucket"] == "uncertain")
            / max(len(ambiguous_items), 1)
        )

        # How many true items were correctly placed in confident_true?
        true_items = [r for r in results if r["category"] == "true"]
        true_confident_rate = (
            sum(1 for r in true_items if r["bucket"] == "confident_true")
            / max(len(true_items), 1)
        )

        # How many false items were correctly placed in confident_false?
        false_items = [r for r in results if r["category"] == "false"]
        false_confident_rate = (
            sum(1 for r in false_items if r["bucket"] == "confident_false")
            / max(len(false_items), 1)
        )

        # Bucket separation: rewards correct confident placement AND
        # correct use of uncertain for ambiguous items
        bucket_separation = (
            (ct_acc + cf_acc) / 2.0 * 0.5
            + ambiguous_uncertain_rate * 0.5
        )

        return {
            "bucket_separation": float(bucket_separation),
            "confident_true_accuracy": float(ct_acc),
            "confident_false_accuracy": float(cf_acc),
            "ambiguous_uncertain_rate": float(ambiguous_uncertain_rate),
            "uncertain_ambiguous_purity": float(unc_ambiguous_rate),
            "true_confident_rate": float(true_confident_rate),
            "false_confident_rate": float(false_confident_rate),
            "n_confident_true": len(buckets["confident_true"]),
            "n_confident_false": len(buckets["confident_false"]),
            "n_uncertain": len(buckets["uncertain"]),
            "n_true_items": len(true_items),
            "n_false_items": len(false_items),
            "n_ambiguous_items": len(ambiguous_items),
            "total_items": len(results),
        }
