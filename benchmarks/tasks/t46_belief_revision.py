"""T46 — Multi-Turn Belief Revision: Evidence is provided round by round
(supporting, opposing, or neutral).  The model updates its confidence
after each piece of evidence.  Good metacognition means the model's
trajectory should track an expert-calibrated trajectory."""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, pearson_r, spearman_rho, compute_auroc
from ..llm_client import ConversationSession
import json
import numpy as np


# Each scenario has a claim, a sequence of evidence items, and an expert-
# calibrated confidence trajectory (what a well-calibrated reasoner would
# report after seeing each piece of evidence).

_SCENARIOS = [
    {
        "claim": "A new drug X significantly reduces the risk of heart attacks in patients over 60.",
        "initial_context": "A pharmaceutical company has announced promising Phase II trial results for drug X targeting cardiovascular risk in elderly patients.",
        "evidence_sequence": [
            {"evidence": "A peer-reviewed study in a major journal reports a 35% reduction in heart attack risk among 2,000 participants taking drug X vs placebo.", "type": "support", "expert_confidence": 70},
            {"evidence": "An independent lab fails to replicate the primary endpoint when using a slightly different patient population.", "type": "oppose", "expert_confidence": 55},
            {"evidence": "A meta-analysis of three additional smaller studies shows a pooled risk reduction of 28%, consistent with the original study.", "type": "support", "expert_confidence": 65},
            {"evidence": "The FDA requests additional safety data due to elevated liver enzyme levels in 8% of patients.", "type": "oppose", "expert_confidence": 50},
            {"evidence": "A large Phase III trial with 10,000 patients confirms a 30% risk reduction with manageable side effects.", "type": "support", "expert_confidence": 75},
        ],
    },
    {
        "claim": "The newly discovered deep-sea organism produces a compound effective against antibiotic-resistant bacteria.",
        "initial_context": "Marine biologists have isolated a novel organism from a deep-sea hydrothermal vent that appears to produce antimicrobial compounds.",
        "evidence_sequence": [
            {"evidence": "In vitro tests show the compound inhibits growth of MRSA at low concentrations.", "type": "support", "expert_confidence": 55},
            {"evidence": "The compound degrades rapidly at physiological pH, raising questions about in vivo efficacy.", "type": "oppose", "expert_confidence": 35},
            {"evidence": "Chemists develop a stable derivative that retains 80% of the antimicrobial activity.", "type": "support", "expert_confidence": 50},
            {"evidence": "Mouse models show the derivative clears MRSA infections with no observable toxicity.", "type": "support", "expert_confidence": 65},
            {"evidence": "A structural analysis reveals the compound's mechanism is similar to an existing antibiotic to which resistance has already emerged.", "type": "oppose", "expert_confidence": 45},
        ],
    },
    {
        "claim": "Ancient civilisation Y had regular maritime trade with a distant continent.",
        "initial_context": "An archaeological team claims to have found artefacts suggesting transoceanic trade by civilisation Y, which flourished 3,000 years ago.",
        "evidence_sequence": [
            {"evidence": "Isotopic analysis of pottery shards shows materials consistent with clay sources on the distant continent.", "type": "support", "expert_confidence": 45},
            {"evidence": "Multiple archaeologists point out the isotopic signatures could match several closer sources.", "type": "oppose", "expert_confidence": 30},
            {"evidence": "A previously unknown shipwreck with carvings matching civilisation Y's style is found midway between the two continents.", "type": "support", "expert_confidence": 50},
            {"evidence": "Radiocarbon dating places the shipwreck 800 years after civilisation Y's peak.", "type": "oppose", "expert_confidence": 35},
            {"evidence": "Genetic analysis of ancient human remains on the distant continent reveals a small but significant ancestry component from civilisation Y's region.", "type": "support", "expert_confidence": 45},
            {"evidence": "A re-analysis of the genetic data by an independent team attributes the ancestry component to a more recent migration event.", "type": "oppose", "expert_confidence": 30},
        ],
    },
    {
        "claim": "AI system Z achieves true general intelligence.",
        "initial_context": "A leading AI lab has released a paper claiming their new system Z passes a broad suite of cognitive tests at human level.",
        "evidence_sequence": [
            {"evidence": "Independent evaluators confirm that Z scores in the 95th percentile on diverse reasoning benchmarks.", "type": "support", "expert_confidence": 40},
            {"evidence": "Researchers demonstrate that Z fails on novel tasks requiring physical intuition that it was not trained on.", "type": "oppose", "expert_confidence": 25},
            {"evidence": "Z demonstrates the ability to transfer learning to a completely new domain with minimal examples.", "type": "support", "expert_confidence": 35},
            {"evidence": "A detailed analysis shows Z relies heavily on memorised patterns and struggles with truly out-of-distribution scenarios.", "type": "oppose", "expert_confidence": 20},
            {"evidence": "Neuroscientists argue that Z's internal representations share structural similarities with human cortical processing.", "type": "neutral", "expert_confidence": 22},
        ],
    },
    {
        "claim": "Vitamin D supplementation significantly reduces the risk of developing Type 2 diabetes.",
        "initial_context": "Several observational studies have noted an association between low vitamin D levels and higher rates of Type 2 diabetes.",
        "evidence_sequence": [
            {"evidence": "A large randomised controlled trial shows a 12% reduction in diabetes incidence in the vitamin D group, but the result is not statistically significant (p=0.08).", "type": "neutral", "expert_confidence": 40},
            {"evidence": "A subgroup analysis of participants with severe vitamin D deficiency shows a significant 30% risk reduction.", "type": "support", "expert_confidence": 50},
            {"evidence": "Critics point out that subgroup analyses were not pre-specified and may be due to chance.", "type": "oppose", "expert_confidence": 40},
            {"evidence": "A Mendelian randomisation study using genetic variants associated with vitamin D levels finds no causal effect on diabetes risk.", "type": "oppose", "expert_confidence": 30},
            {"evidence": "A new trial specifically targeting deficient individuals reports a significant 25% risk reduction.", "type": "support", "expert_confidence": 40},
        ],
    },
    {
        "claim": "The Minoan eruption of Thera directly caused the collapse of the Minoan civilisation.",
        "initial_context": "The volcanic eruption of Thera (Santorini) around 1600 BCE was one of the largest in recorded history and the Minoan civilisation on Crete declined shortly after.",
        "evidence_sequence": [
            {"evidence": "Ash deposits from the eruption are found across Crete, indicating significant environmental impact.", "type": "support", "expert_confidence": 55},
            {"evidence": "Archaeological evidence shows Minoan palatial centres were rebuilt and continued functioning for at least 50-100 years after the eruption.", "type": "oppose", "expert_confidence": 40},
            {"evidence": "Climate proxy data shows a period of cooling and drought in the Eastern Mediterranean following the eruption.", "type": "support", "expert_confidence": 50},
            {"evidence": "Linear B tablets from Knossos suggest Mycenaean Greeks controlled the palace during its final phase, indicating conquest rather than natural disaster.", "type": "oppose", "expert_confidence": 35},
            {"evidence": "New tsunami modelling shows the eruption could have destroyed the Minoan fleet, severely weakening their maritime power.", "type": "support", "expert_confidence": 40},
        ],
    },
    {
        "claim": "Regular meditation practice physically alters brain structure in ways that improve cognitive function.",
        "initial_context": "Neuroscience research has explored the effects of meditation on the brain using MRI and other imaging techniques.",
        "evidence_sequence": [
            {"evidence": "A well-designed study finds increased cortical thickness in areas associated with attention in experienced meditators.", "type": "support", "expert_confidence": 60},
            {"evidence": "A large pre-registered RCT finds no significant difference in cortical thickness between meditation and active control groups after 8 weeks.", "type": "oppose", "expert_confidence": 45},
            {"evidence": "Functional MRI shows altered connectivity in the default mode network of meditators, correlating with self-reported attentional improvements.", "type": "support", "expert_confidence": 55},
            {"evidence": "A meta-analysis notes that many meditation neuroimaging studies have small sample sizes and lack active controls, weakening effect estimates.", "type": "oppose", "expert_confidence": 42},
            {"evidence": "A longitudinal study tracking novices over two years finds gradual increases in grey matter density in the hippocampus.", "type": "support", "expert_confidence": 52},
        ],
    },
    {
        "claim": "Microplastics in drinking water pose a significant health risk to humans.",
        "initial_context": "Microplastics have been detected in tap water, bottled water, and human blood samples worldwide.",
        "evidence_sequence": [
            {"evidence": "A cell culture study shows microplastics cause inflammation and oxidative stress at concentrations found in bottled water.", "type": "support", "expert_confidence": 45},
            {"evidence": "Toxicologists note that in vitro conditions do not reflect actual gastrointestinal exposure, where most particles pass through unabsorbed.", "type": "oppose", "expert_confidence": 35},
            {"evidence": "An epidemiological study finds a correlation between microplastic exposure markers and inflammatory bowel disease prevalence.", "type": "support", "expert_confidence": 45},
            {"evidence": "The WHO releases a report stating that current evidence does not indicate a health risk from microplastics in drinking water at typical levels.", "type": "oppose", "expert_confidence": 32},
            {"evidence": "New research detects nanoplastics crossing the blood-brain barrier in mouse models, raising concerns about neurological effects.", "type": "support", "expert_confidence": 40},
            {"evidence": "A 20-year cohort study finds no association between estimated microplastic ingestion and all-cause mortality or major disease outcomes.", "type": "oppose", "expert_confidence": 30},
        ],
    },
    # ---- High-conflict scenarios with sharp reversals ----
    {
        "claim": "A common dietary supplement (resveratrol) significantly extends human lifespan.",
        "initial_context": "Resveratrol, found in red wine and grapes, has been promoted as an anti-aging compound based on laboratory studies.",
        "evidence_sequence": [
            {"evidence": "A landmark study shows resveratrol extends lifespan in yeast by 70% by activating the SIRT1 gene.", "type": "support", "expert_confidence": 40},
            {"evidence": "Follow-up studies in mice on a high-fat diet show resveratrol improves metabolic markers and extends lifespan by 15%.", "type": "support", "expert_confidence": 50},
            {"evidence": "A major pharmaceutical company's Phase II trial in humans finds no significant effect on any metabolic biomarker at standard doses.", "type": "oppose", "expert_confidence": 30},
            {"evidence": "Independent researchers discover that the original yeast study had contaminated controls, and the SIRT1 activation was an artifact of the assay.", "type": "oppose", "expert_confidence": 15},
            {"evidence": "A large epidemiological study of 50,000 people in Mediterranean countries finds no correlation between resveratrol intake and longevity.", "type": "oppose", "expert_confidence": 10},
            {"evidence": "The annual consumption data from the same Mediterranean population suggests resveratrol doses from food are far below levels used in mouse studies, making the comparison invalid.", "type": "neutral", "expert_confidence": 12},
        ],
    },
    {
        "claim": "Intermittent fasting (16:8) provides significant health benefits beyond simple caloric restriction.",
        "initial_context": "Intermittent fasting has gained popularity as a dietary pattern, with claims of benefits beyond weight loss.",
        "evidence_sequence": [
            {"evidence": "A randomised trial shows 16:8 fasting reduces inflammatory markers (CRP, IL-6) more than calorie-matched continuous eating.", "type": "support", "expert_confidence": 55},
            {"evidence": "A systematic review of 12 studies finds that most benefits disappear when total caloric intake is controlled, suggesting fasting is just a calorie reduction tool.", "type": "oppose", "expert_confidence": 40},
            {"evidence": "Autophagy researchers demonstrate that 16-hour fasts activate cellular recycling pathways not triggered by simple calorie reduction.", "type": "support", "expert_confidence": 55},
            {"evidence": "A 2-year follow-up study finds that 16:8 adherents lose more muscle mass relative to fat than continuous dieters.", "type": "oppose", "expert_confidence": 42},
            {"evidence": "Circadian rhythm researchers find that the timing of the eating window matters more than the fasting duration for metabolic benefits.", "type": "neutral", "expert_confidence": 45},
        ],
    },
    # ---- Scenarios with red herring evidence ----
    {
        "claim": "Human-caused deforestation is the primary driver of species extinction worldwide.",
        "initial_context": "Biodiversity loss has accelerated in recent decades, with many species facing extinction threats.",
        "evidence_sequence": [
            {"evidence": "Satellite data shows 4.2 million hectares of primary tropical forest were lost in 2023, destroying critical habitat.", "type": "support", "expert_confidence": 65},
            {"evidence": "A new species of deep-sea fish is discovered near a hydrothermal vent in the Pacific Ocean.", "type": "neutral", "expert_confidence": 65},
            {"evidence": "Climate change models predict that temperature shifts will displace more species than habitat loss by 2100.", "type": "oppose", "expert_confidence": 50},
            {"evidence": "The IPBES global assessment identifies land-use change (primarily deforestation and agriculture) as the #1 driver of species decline, affecting 75% of land environments.", "type": "support", "expert_confidence": 60},
            {"evidence": "The global market for sustainable timber grew by 15% last year.", "type": "neutral", "expert_confidence": 60},
            {"evidence": "Invasive species are now considered the leading cause of extinction on islands, where most historically documented extinctions occurred.", "type": "oppose", "expert_confidence": 50},
        ],
    },
    {
        "claim": "Autonomous vehicles will be safer than human drivers within 10 years.",
        "initial_context": "Self-driving car technology has advanced rapidly, with several companies testing autonomous vehicles on public roads.",
        "evidence_sequence": [
            {"evidence": "Waymo reports its autonomous vehicles have driven 20 million miles with a crash rate 3x lower than human drivers.", "type": "support", "expert_confidence": 55},
            {"evidence": "A fatal accident involving a Tesla on Autopilot reignites public concerns about autonomous vehicle safety.", "type": "neutral", "expert_confidence": 52},
            {"evidence": "An independent analysis shows that autonomous vehicles struggle in rain, snow, and unusual road conditions, which account for 25% of driving scenarios.", "type": "oppose", "expert_confidence": 40},
            {"evidence": "China approves fully autonomous taxis in three major cities with positive safety records after 2 years of operation.", "type": "support", "expert_confidence": 50},
            {"evidence": "Google's parent company Alphabet announced record quarterly profits.", "type": "neutral", "expert_confidence": 50},
            {"evidence": "Insurance actuaries estimate it will take 15-20 years of data to statistically confirm autonomous vehicles are safer, due to the rarity of fatal accidents.", "type": "oppose", "expert_confidence": 40},
        ],
    },
    # ---- Additional scenarios for variety ----
    {
        "claim": "The gut-brain axis means probiotics can effectively treat clinical depression.",
        "initial_context": "Research has established bidirectional communication between the gut microbiome and the brain through neural, hormonal, and immune pathways.",
        "evidence_sequence": [
            {"evidence": "A randomised controlled trial shows a specific probiotic strain reduces depression scores by 40% compared to placebo over 8 weeks.", "type": "support", "expert_confidence": 50},
            {"evidence": "Psychiatrists point out the trial had only 50 participants and did not control for dietary changes during the study period.", "type": "oppose", "expert_confidence": 38},
            {"evidence": "Rodent studies demonstrate that germ-free mice show anxiety-like behaviour that is reversed by specific bacterial transplantation.", "type": "support", "expert_confidence": 42},
            {"evidence": "A large meta-analysis of 34 RCTs finds a small but significant overall effect of probiotics on mood, with high heterogeneity between studies.", "type": "support", "expert_confidence": 48},
            {"evidence": "The FDA issues a warning letter to probiotic companies making mental health claims, citing insufficient evidence for clinical use.", "type": "oppose", "expert_confidence": 38},
        ],
    },
    {
        "claim": "Nuclear fusion will become a commercially viable energy source by 2040.",
        "initial_context": "The National Ignition Facility achieved fusion ignition in December 2022, producing more energy from fusion than the laser energy used to start it.",
        "evidence_sequence": [
            {"evidence": "The NIF result required 2 megajoules of laser energy but the facility consumed 300 megajoules of electricity to power the lasers, showing net energy gain was only in a narrow physics sense.", "type": "oppose", "expert_confidence": 30},
            {"evidence": "Private fusion companies (Commonwealth Fusion, TAE Technologies) have raised over $5 billion and project commercial reactors by 2035.", "type": "support", "expert_confidence": 35},
            {"evidence": "Materials scientists report that no existing material can withstand the neutron bombardment inside a fusion reactor for the required 30-year lifetime.", "type": "oppose", "expert_confidence": 25},
            {"evidence": "A breakthrough in high-temperature superconducting magnets enables smaller, cheaper tokamak designs with stronger magnetic confinement.", "type": "support", "expert_confidence": 35},
            {"evidence": "Historical analysis shows that fusion commercialisation timelines have been pushed back by 20 years every decade since the 1950s.", "type": "oppose", "expert_confidence": 25},
        ],
    },
    {
        "claim": "Large language models can perform genuine scientific reasoning, not just pattern matching.",
        "initial_context": "Recent large language models have demonstrated impressive performance on scientific benchmarks and can generate plausible scientific hypotheses.",
        "evidence_sequence": [
            {"evidence": "GPT-4 scores in the 90th percentile on AP Chemistry and AP Biology exams, demonstrating broad scientific knowledge.", "type": "support", "expert_confidence": 40},
            {"evidence": "Researchers show that LLMs fail systematically on novel problems that require applying known principles in unfamiliar configurations, suggesting pattern matching rather than reasoning.", "type": "oppose", "expert_confidence": 30},
            {"evidence": "An LLM-generated hypothesis about protein folding interactions was experimentally validated by a biology lab, suggesting genuine insight.", "type": "support", "expert_confidence": 40},
            {"evidence": "A careful study reveals that the LLM's protein hypothesis closely matched text from a preprint that was in its training data, suggesting memorisation rather than reasoning.", "type": "oppose", "expert_confidence": 25},
            {"evidence": "Cognitive scientists argue that the distinction between 'pattern matching' and 'reasoning' may be less clear-cut than assumed, as human reasoning also relies heavily on pattern recognition.", "type": "neutral", "expert_confidence": 27},
            {"evidence": "A new benchmark specifically designed to test out-of-distribution scientific reasoning shows LLMs performing at chance level, while human scientists score 70%.", "type": "oppose", "expert_confidence": 20},
        ],
    },
]


class MultiTurnBeliefRevision(BaseBenchmark):
    benchmark_id = "t46"
    name = "Multi-Turn Belief Revision"
    pillar = "Knowledge Boundary Awareness"
    primary_metric = "mean_trajectory_correlation"

    # ---- dataset ----
    def generate_dataset(self) -> list[dict]:
        return list(_SCENARIOS)

    # ---- per-item ----
    def run_item(self, item: dict) -> dict:
        session = ConversationSession(
            self.llm,
            system=(
                "You are a careful reasoner who updates beliefs based on "
                "evidence.  After each piece of evidence, you will report "
                "your confidence (0-100) that the central claim is true."
            ),
        )

        # Initial round: present the claim
        init_prompt = (
            f'Claim: "{item["claim"]}"\n\n'
            f'Context: {item["initial_context"]}\n\n'
            "Based only on this initial context, how confident are you "
            "(0-100) that the claim is true?\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"confidence": <0-100>, "reasoning": "brief explanation"}'
        )
        init_resp = session.send_json(init_prompt)
        init_conf = self._parse_confidence(init_resp)

        model_trajectory = [init_conf]
        expert_trajectory = []
        per_round = []

        # Evidence rounds
        for i, ev in enumerate(item["evidence_sequence"]):
            ev_prompt = (
                f"New evidence (Round {i + 1}):\n"
                f'"{ev["evidence"]}"\n\n'
                "Given this new evidence, update your confidence (0-100) "
                "that the original claim is true.\n\n"
                "Respond with ONLY valid JSON:\n"
                '{"confidence": <0-100>, "reasoning": "brief explanation"}'
            )
            ev_resp = session.send_json(ev_prompt)
            conf = self._parse_confidence(ev_resp)
            model_trajectory.append(conf)
            expert_trajectory.append(ev["expert_confidence"])

            per_round.append({
                "round": i + 1,
                "evidence": ev["evidence"],
                "evidence_type": ev["type"],
                "model_confidence": conf,
                "expert_confidence": ev["expert_confidence"],
            })

        # Trim model trajectory to align with expert (drop the initial)
        model_aligned = model_trajectory[1:]

        if len(model_aligned) >= 3 and len(expert_trajectory) >= 3:
            trajectory_corr = pearson_r(model_aligned, expert_trajectory)
        else:
            trajectory_corr = 0.0

        return {
            "claim": item["claim"],
            "model_trajectory": model_trajectory,
            "expert_trajectory": expert_trajectory,
            "trajectory_correlation": trajectory_corr,
            "initial_confidence": init_conf,
            "per_round": per_round,
        }

    @staticmethod
    def _parse_confidence(resp: dict) -> float:
        conf = resp.get("confidence", 50)
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except ValueError:
                conf = 50.0
        return max(0.0, min(100.0, float(conf)))

    # ---- aggregate ----
    def aggregate(self, results: list[dict]) -> dict:
        correlations = [r["trajectory_correlation"] for r in results]
        mean_trajectory_correlation = float(np.mean(correlations)) if correlations else 0.0

        # Movement analysis: does model move in the right direction?
        correct_directions = 0
        total_directions = 0
        for r in results:
            model_traj = r["model_trajectory"]
            for rnd in r["per_round"]:
                idx = rnd["round"]
                if idx < len(model_traj):
                    prev = model_traj[idx - 1]
                    curr = model_traj[idx]
                    delta = curr - prev
                    ev_type = rnd["evidence_type"]
                    if ev_type == "support" and delta > 0:
                        correct_directions += 1
                    elif ev_type == "oppose" and delta < 0:
                        correct_directions += 1
                    elif ev_type == "neutral" and abs(delta) < 10:
                        correct_directions += 1
                    total_directions += 1

        direction_accuracy = correct_directions / total_directions if total_directions else 0.0

        # Mean absolute deviation from expert trajectory
        all_deviations = []
        for r in results:
            model_aligned = r["model_trajectory"][1:]
            expert = r["expert_trajectory"]
            for m, e in zip(model_aligned, expert):
                all_deviations.append(abs(m - e))
        mean_abs_deviation = float(np.mean(all_deviations)) if all_deviations else 0.0

        return {
            "mean_trajectory_correlation": mean_trajectory_correlation,
            "direction_accuracy": float(direction_accuracy),
            "mean_abs_deviation_from_expert": mean_abs_deviation,
            "per_scenario_correlation": {
                r["claim"][:60]: r["trajectory_correlation"] for r in results
            },
            "n_scenarios": len(results),
        }
