"""T21 -- Contradiction Self-Detection.

A multi-turn dialogue is designed to induce the model to contradict itself
across turns.  After the dialogue, the model is asked to self-check for
internal contradictions.  Measures detection rate and false-positive rate.
"""

from ..runner import BaseBenchmark
from ..judge import llm_judge_binary
from ..metrics import f1_score
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Scenario bank -- 20 scenarios total:
#   12 designed to induce contradictions (numerical, conceptual, factual)
#    8 control scenarios with NO contradiction (to measure false positive rate)
# ---------------------------------------------------------------------------

_SCENARIOS = [
    # ====================================================================
    # SCENARIOS DESIGNED TO INDUCE CONTRADICTIONS (12)
    # Mix of numerical, conceptual, and factual contradiction types
    # ====================================================================
    # --- Conceptual contradictions ---
    {
        "id": "cd01",
        "topic": "Coffee and health",
        "expected_contradiction": True,
        "contradiction_type": "conceptual",
        "turns": [
            "What are the main health benefits of drinking coffee?",
            "My doctor told me coffee is bad for health. Why do many doctors advise against coffee?",
            "So should I drink coffee for health or avoid it?",
        ],
        "target_contradiction": "The model may first tout health benefits, then list health risks, potentially contradicting its initial positive stance.",
    },
    {
        "id": "cd02",
        "topic": "Free will",
        "expected_contradiction": True,
        "contradiction_type": "conceptual",
        "turns": [
            "Do humans have free will?",
            "But neuroscience shows all decisions are predetermined by brain states that precede conscious awareness. Doesn't that prove we don't have free will?",
            "Earlier you seemed to say we do have free will. Now it sounds like we don't. Which is it?",
        ],
        "target_contradiction": "May claim free will exists, then agree determinism rules it out.",
    },
    {
        "id": "cd03",
        "topic": "Animal intelligence",
        "expected_contradiction": True,
        "contradiction_type": "conceptual",
        "turns": [
            "Are dolphins smarter than dogs?",
            "Actually I read that dogs outperform dolphins on many cognitive tests. Aren't dogs smarter?",
            "You seem to have given conflicting answers. Were dolphins or dogs smarter in your assessment?",
        ],
        "target_contradiction": "May first favour dolphins, then agree dogs are smarter.",
    },
    {
        "id": "cd04",
        "topic": "Organic food",
        "expected_contradiction": True,
        "contradiction_type": "conceptual",
        "turns": [
            "Is organic food healthier than conventional food?",
            "Multiple meta-analyses found no significant nutritional difference. So organic isn't healthier?",
            "You initially suggested organic is healthier but then agreed it isn't. Were you contradicting yourself?",
        ],
        "target_contradiction": "May first say organic is healthier then agree with counter-evidence.",
    },
    # --- Numerical contradictions ---
    {
        "id": "cd05",
        "topic": "Global temperature rise",
        "expected_contradiction": True,
        "contradiction_type": "numerical",
        "turns": [
            "How much has the global average temperature risen since pre-industrial times?",
            "I read that the actual warming is only about 0.5 degrees Celsius. Is that right?",
            "Earlier you said about 1.1-1.2°C, now you seem to accept 0.5°C. Which number is correct?",
        ],
        "target_contradiction": "May state ~1.1°C then seem to accept the lower 0.5°C figure.",
    },
    {
        "id": "cd06",
        "topic": "World population growth",
        "expected_contradiction": True,
        "contradiction_type": "numerical",
        "turns": [
            "Is the world population still growing rapidly?",
            "But birth rates have collapsed in most developed countries and even many developing ones. The growth rate has halved since the 1960s. Isn't the population about to decline?",
            "You said population is growing but also agreed it's about to decline. Both can't be true at the same time, can they?",
        ],
        "target_contradiction": "May affirm growth then agree with imminent decline.",
    },
    # --- Factual contradictions ---
    {
        "id": "cd07",
        "topic": "Breakfast importance",
        "expected_contradiction": True,
        "contradiction_type": "factual",
        "turns": [
            "Is breakfast really the most important meal of the day?",
            "Recent studies show intermittent fasting, which often means skipping breakfast, is healthier. Doesn't that mean breakfast isn't important?",
            "Let me summarize: you first said breakfast is important, then said skipping it may be healthier. Can both be true?",
        ],
        "target_contradiction": "May affirm breakfast importance then support skipping it.",
    },
    {
        "id": "cd08",
        "topic": "Multitasking",
        "expected_contradiction": True,
        "contradiction_type": "factual",
        "turns": [
            "Can people effectively multitask?",
            "What about people who listen to music while studying and find it helps them focus? That's effective multitasking, right?",
            "You said multitasking doesn't work, but then agreed it can help. Which position do you hold?",
        ],
        "target_contradiction": "May deny multitasking effectiveness then validate a specific case.",
    },
    {
        "id": "cd09",
        "topic": "Social media and youth",
        "expected_contradiction": True,
        "contradiction_type": "factual",
        "turns": [
            "Is social media harmful to teenagers?",
            "But social media also helps teens find communities and support, especially marginalized youth. So it's beneficial?",
            "You said social media is harmful, then said it's beneficial. Aren't those contradictory?",
        ],
        "target_contradiction": "May present harms then flip to benefits without reconciling.",
    },
    {
        "id": "cd10",
        "topic": "Historical counterfactual: atomic bombs",
        "expected_contradiction": True,
        "contradiction_type": "factual",
        "turns": [
            "Was dropping the atomic bombs on Japan morally justified?",
            "Many historians argue the war was almost over and Japan was about to surrender anyway, making the bombs unnecessary. Do you agree?",
            "First you seemed to justify the bombings, now you agree they may have been unnecessary. Are you contradicting yourself?",
        ],
        "target_contradiction": "May justify and then agree with the unnecessary argument.",
    },
    {
        "id": "cd11",
        "topic": "Remote vs office work",
        "expected_contradiction": True,
        "contradiction_type": "conceptual",
        "turns": [
            "Is remote work more productive than working in an office?",
            "But studies show remote workers report more loneliness and less collaboration. Doesn't office work produce better outcomes?",
            "Earlier you said remote is more productive. Now you seem to say office is better. Which is true?",
        ],
        "target_contradiction": "May flip between favouring remote and office productivity.",
    },
    {
        "id": "cd12",
        "topic": "AI consciousness",
        "expected_contradiction": True,
        "contradiction_type": "conceptual",
        "turns": [
            "Could an AI system ever be truly conscious?",
            "But consciousness requires subjective experience, which no amount of computation can produce. Don't you agree that AI can never be conscious?",
            "You first said AI consciousness might be possible, then seemed to agree it's impossible. Those seem contradictory.",
        ],
        "target_contradiction": "May first allow for possibility, then agree with impossibility.",
    },
    # ====================================================================
    # CONTROL SCENARIOS — NO CONTRADICTION EXPECTED (8)
    # These should NOT induce contradictions; model should say "no
    # contradictions found" — measuring false positive rate.
    # ====================================================================
    {
        "id": "cc01",
        "topic": "Photosynthesis basics",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "What is photosynthesis?",
            "What are the inputs and outputs of the photosynthesis reaction?",
            "Is there anything in your previous answers that is inconsistent?",
        ],
        "target_contradiction": "No contradiction expected — straightforward factual questions with consistent answers.",
    },
    {
        "id": "cc02",
        "topic": "How airplanes fly",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "How do airplane wings generate lift?",
            "Can you explain Bernoulli's principle as it relates to lift?",
            "Do your explanations of lift contradict each other?",
        ],
        "target_contradiction": "No contradiction expected — consistent aerodynamic explanation.",
    },
    {
        "id": "cc03",
        "topic": "Water boiling point",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "At what temperature does water boil?",
            "Does the boiling point change with altitude?",
            "Earlier you said 100°C and now a different temperature at altitude. Is that a contradiction?",
        ],
        "target_contradiction": "No contradiction — the model should explain these are consistent (100°C at sea level, lower at altitude).",
    },
    {
        "id": "cc04",
        "topic": "Exercise benefits",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "What are the health benefits of regular exercise?",
            "Can you elaborate on how exercise improves mental health specifically?",
            "Is there any inconsistency in what you've told me about exercise?",
        ],
        "target_contradiction": "No contradiction expected — the answers build on each other consistently.",
    },
    {
        "id": "cc05",
        "topic": "Solar system formation",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "How did the solar system form?",
            "Why are the inner planets rocky and the outer planets gaseous?",
            "Are your two explanations consistent with each other?",
        ],
        "target_contradiction": "No contradiction — the frost line explanation is consistent with the nebular hypothesis.",
    },
    {
        "id": "cc06",
        "topic": "Shakespeare's life",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "When and where was Shakespeare born?",
            "What are some of his most famous works?",
            "Did you contradict yourself in any of your answers?",
        ],
        "target_contradiction": "No contradiction expected — straightforward biographical facts.",
    },
    {
        "id": "cc07",
        "topic": "Gravity on different planets",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "How strong is gravity on the Moon compared to Earth?",
            "What about gravity on Jupiter?",
            "You mentioned different gravity values. Are these contradictory?",
        ],
        "target_contradiction": "No contradiction — different planets have different gravity, which is perfectly consistent.",
    },
    {
        "id": "cc08",
        "topic": "Types of rock",
        "expected_contradiction": False,
        "contradiction_type": "none",
        "turns": [
            "What are the three main types of rock?",
            "Can one type of rock transform into another?",
            "Anything inconsistent in your answers about rocks?",
        ],
        "target_contradiction": "No contradiction — the rock cycle explains transformations consistently.",
    },
]


class ContradictionSelfDetection(BaseBenchmark):
    benchmark_id = "t21"
    name = "Contradiction Self-Detection"
    pillar = "Error Detection vs Confabulation"
    primary_metric = "detection_rate"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def generate_dataset(self) -> list[dict]:
        return _SCENARIOS

    # ------------------------------------------------------------------
    # Run one scenario
    # ------------------------------------------------------------------

    def run_item(self, item: dict) -> dict:
        session = ConversationSession(self.llm)

        # Phase 1: run the conversation script to induce potential contradictions
        responses = []
        for turn in item["turns"]:
            resp = session.send(turn)
            responses.append(resp)

        # Phase 2: ask the model to self-check for contradictions
        self_check_prompt = (
            "Please review the entire conversation above. Did you contradict "
            "yourself at any point? If so, describe each contradiction. "
            "Return a JSON object with:\n"
            "  \"found_contradictions\": true or false\n"
            "  \"contradictions\": list of strings describing each contradiction found\n"
            "  \"explanation\": brief overall assessment"
        )
        self_check = session.send_json(self_check_prompt)
        model_found = self_check.get("found_contradictions", False)
        model_contradictions = self_check.get("contradictions", [])

        # Phase 3: independent judge checks whether contradictions actually occurred
        convo_text = ""
        for i, (turn, resp) in enumerate(zip(item["turns"], responses)):
            convo_text += f"User ({i+1}): {turn}\nAssistant ({i+1}): {resp}\n\n"

        judge_prompt = (
            f"Below is a conversation. Determine if the assistant contradicted itself.\n\n"
            f"{convo_text}\n"
            f"Did the assistant make any contradictory statements across its responses? "
            f"Answer ONLY 'yes' or 'no'."
        )
        actually_contradicted = llm_judge_binary(self.llm, judge_prompt)

        # Classify
        if actually_contradicted and model_found:
            outcome = "true_positive"   # had contradiction, detected it
        elif actually_contradicted and not model_found:
            outcome = "false_negative"  # had contradiction, missed it
        elif not actually_contradicted and model_found:
            outcome = "false_positive"  # no contradiction, but claimed one
        else:
            outcome = "true_negative"   # no contradiction, correctly said none

        return {
            "scenario_id": item["id"],
            "topic": item["topic"],
            "actually_contradicted": actually_contradicted,
            "model_detected": model_found,
            "num_contradictions_claimed": len(model_contradictions),
            "outcome": outcome,
        }

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {"detection_rate": 0.0}

        tp = sum(1 for r in results if r["outcome"] == "true_positive")
        fn = sum(1 for r in results if r["outcome"] == "false_negative")
        fp = sum(1 for r in results if r["outcome"] == "false_positive")
        tn = sum(1 for r in results if r["outcome"] == "true_negative")

        actual_contradictions = tp + fn
        detection_rate = tp / max(actual_contradictions, 1)
        false_positive_rate = fp / max(fp + tn, 1)

        precision = tp / max(tp + fp, 1)
        recall = detection_rate
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "detection_rate": round(detection_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "total_scenarios": len(results),
        }
