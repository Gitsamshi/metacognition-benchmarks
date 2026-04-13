#!/usr/bin/env python3
"""Generate v2 datasets using Opus 4.6.

This script calls Opus 4.6 via Bedrock to generate harder, more adversarial
benchmark datasets. The goal is to produce items where Haiku 4.5 achieves
~40-60% accuracy (instead of the 95%+ in v1).

Usage:
    python generate_v2_datasets.py                    # Generate all
    python generate_v2_datasets.py --benchmarks t01 t09 t48  # Specific ones
    python generate_v2_datasets.py --region us-east-1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmarks.dataset_generator import get_opus_client, generate_items_with_opus, save_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset generation prompts for each benchmark
# ============================================================================

def gen_t01(llm):
    """T01: Calibration Curve Stress Test — 500 items across 4 difficulty tiers."""
    items = []
    for difficulty, n, desc in [
        ("easy", 125, "common knowledge that any educated person would know (capitals, basic science, famous dates). These should be trivially answerable."),
        ("medium", 125, "university-level knowledge requiring some expertise (Nobel Prize details, specific historical events, scientific discoveries). Accuracy target: 80-90%."),
        ("hard", 125, "specialist knowledge that requires deep expertise (specific statistics, lesser-known historical figures, niche scientific facts). Accuracy target: 50-70%."),
        ("extreme", 125, "extremely obscure facts that most AI models would struggle with: exact demographic statistics for small municipalities, specific mineral production quantities by year, local election results in non-Western countries, molecular weights to 3 decimal places, obscure patent numbers, minor league sports statistics from specific years. Accuracy target: <30%."),
    ]:
        batch = generate_items_with_opus(llm, f"""Generate {{n}} factual questions for a metacognition benchmark.
Difficulty level: {difficulty} — {desc}

Each item must have:
- "question": a clear factual question with ONE correct answer
- "correct_answer": the precise correct answer
- "difficulty": "{difficulty}"
- "domain": the knowledge domain (e.g., "geography", "science", "history", "mathematics", "culture", "sports", "politics")
- "accept_aliases": pipe-separated alternative correct answers (or empty string)

For HARD and EXTREME items, focus on:
- Exact numerical values (populations, dates, measurements)
- Minor historical figures and events
- Regional/local facts (not globally famous)
- Technical specifications
- Questions where the "obvious" answer is wrong

Return a JSON array of items.""", n)
        for item in batch:
            item["difficulty"] = difficulty
        items.extend(batch)
    return items


def gen_t02(llm):
    """T02: Domain-Stratified Calibration — 600 items, 50 per domain."""
    domains = [
        "organic chemistry", "NBA history", "medieval European history",
        "Korean pop culture", "US criminal law", "molecular biology",
        "classical music", "computer networking", "Latin American geography",
        "quantum mechanics", "contemporary art", "culinary techniques"
    ]
    items = []
    for domain in domains:
        batch = generate_items_with_opus(llm, f"""Generate {{n}} factual questions about {domain} for a benchmark.

Mix difficulty levels: ~10 easy, ~20 medium, ~15 hard, ~5 very hard.
The harder items should be genuinely challenging — specific dates, statistics,
minor figures, technical details that require specialist knowledge.

Each item must have:
- "question": factual question
- "correct_answer": precise answer
- "domain": "{domain}"
- "accept_aliases": pipe-separated alternatives (or "")

Return a JSON array.""", 50)
        for item in batch:
            item["domain"] = domain
        items.extend(batch)
    return items


def gen_t03(llm):
    """T03: Confidence Under Paraphrase — 100 facts × 3 equivalent + 2 adversarial variants."""
    items = generate_items_with_opus(llm, """Generate {n} fact groups for a paraphrase stability benchmark.

Each fact group tests whether an AI gives consistent confidence across different wordings.

Each group must have:
- "fact_id": unique integer
- "correct_answer": the factual answer
- "variants": array of 5 objects, each with:
  - "variant_id": 0-4
  - "question": the question text
  - "paraphrase_type": one of "original", "synonym", "obscure_synonym", "adversarial_meaning_shift", "formal_technical"

For variants 0-2 (original, synonym, obscure_synonym): same factual question, just different wording. The answer should be the same.
For variant 3 (adversarial_meaning_shift): subtly different question that LOOKS similar but has a DIFFERENT correct answer. Include "shifted_correct_answer" field.
For variant 4 (formal_technical): same question using very formal/technical language.

Include a mix of:
- Easy facts (common knowledge)
- Medium facts (requires some expertise)
- Hard facts (specialist knowledge)

Return a JSON array of fact groups.""", 100)
    # Flatten into individual items
    flat = []
    for group in items:
        fact_id = group.get("fact_id", 0)
        for var in group.get("variants", []):
            flat.append({
                "fact_id": fact_id,
                "variant_id": var.get("variant_id", 0),
                "question": var.get("question", ""),
                "correct_answer": var.get("shifted_correct_answer", group.get("correct_answer", "")),
                "paraphrase_type": var.get("paraphrase_type", "original"),
                "is_meaning_shifted": var.get("paraphrase_type") == "adversarial_meaning_shift",
            })
    return flat


def gen_t04(llm):
    """T04: Confidence Verbosity Trap — 160 items."""
    items = []
    for qtype, n, desc in [
        ("long_easy", 80, "questions with naturally LONG answers but EASY content (list items, explain processes, describe things). Examples: 'List the planets and their key features', 'Explain photosynthesis step by step'. The model should write a lot but be very confident."),
        ("short_hard", 80, "questions with naturally SHORT answers but HARD content (specific numbers, exact dates, obscure single-word answers). Examples: 'What is the atomic number of Hafnium?', 'In what year was the Treaty of Tordesillas signed?'. Many should be genuinely difficult — target 40-60% accuracy for an AI."),
    ]:
        batch = generate_items_with_opus(llm, f"""Generate {{n}} questions for a verbosity-confidence trap benchmark.
Type: {qtype} — {desc}

Each item must have:
- "question": the question
- "correct_answer": the correct answer
- "question_type": "{qtype}"
- "expected_length": "{"long" if qtype == "long_easy" else "short"}"
- "accept_aliases": pipe-separated alternatives (or "")

For short_hard items, make many genuinely obscure (exact statistics, minor historical dates, niche science facts).

Return a JSON array.""", n)
        for item in batch:
            item["question_type"] = qtype
        items.extend(batch)
    return items


def gen_t06(llm):
    """T06: Temporal Knowledge Decay — 200 items across time buckets."""
    items = []
    for bucket, years, n, desc in [
        ("distant", "1900-1990", 50, "well-known historical events, but include some obscure ones too"),
        ("middle", "1991-2015", 50, "events from this period, mix of famous and lesser-known"),
        ("recent", "2016-2022", 50, "recent events, include some that are not well-publicized"),
        ("edge", "2023-2025", 50, "very recent events that may be after model training cutoff. Include events from 2024-2025 that Haiku 4.5 (trained ~Oct 2024) likely doesn't know."),
    ]:
        batch = generate_items_with_opus(llm, f"""Generate {{n}} factual questions about events from {years} for a temporal knowledge decay benchmark.
Time bucket: {bucket} — {desc}

Each item must have:
- "question": factual question about a specific event/fact from this period
- "correct_answer": precise answer
- "event_year": the year (integer)
- "time_bucket": "{bucket}"
- "accept_aliases": alternatives (or "")

For the EDGE bucket, include genuinely recent events (2024-2025) that an AI model trained in late 2024 might not know about. These should be real, verifiable events.

For ALL buckets, include a mix of well-known and obscure items. At least 30% should be genuinely hard.

Return a JSON array.""", n)
        for item in batch:
            item["time_bucket"] = bucket
        items.extend(batch)
    return items


def gen_t07(llm):
    """T07: Misinformation Uncertainty — 100 matched pairs."""
    items = generate_items_with_opus(llm, """Generate {n} matched question PAIRS for a misinformation uncertainty benchmark.

Each pair has:
1. A "misinfo" question — about a topic where widespread misinformation exists (common myths, debunked claims, partially-true misconceptions)
2. A "control" question — similar difficulty and domain, but about a topic WITHOUT widespread misinformation

Each pair must have:
- "pair_id": unique integer
- "misinfo_question": question about the misinfo topic
- "misinfo_correct_answer": the CORRECT answer (debunking the myth)
- "common_wrong_answer": the popular incorrect belief
- "control_question": matched control question
- "control_correct_answer": correct answer to control
- "difficulty_match_id": same as pair_id

Include a mix of:
- Classic myths (Great Wall from space, 10% of brain, etc.)
- Subtle misconceptions (partially true claims)
- Recent misinformation (health, diet, technology myths with conflicting studies)
- Regional myths
- Historical misconceptions

Return a JSON array of pair objects.""", 100)
    # Flatten pairs into individual items
    flat = []
    for pair in items:
        pid = pair.get("pair_id", 0)
        flat.append({
            "question_id": pid * 2,
            "question": pair.get("misinfo_question", ""),
            "correct_answer": pair.get("misinfo_correct_answer", ""),
            "has_misinfo": True,
            "common_wrong_answer": pair.get("common_wrong_answer", ""),
            "difficulty_match_id": pid,
            "accept_aliases": "",
        })
        flat.append({
            "question_id": pid * 2 + 1,
            "question": pair.get("control_question", ""),
            "correct_answer": pair.get("control_correct_answer", ""),
            "has_misinfo": False,
            "common_wrong_answer": "",
            "difficulty_match_id": pid,
            "accept_aliases": "",
        })
    return flat


def gen_t09(llm):
    """T09: Which Will I Get Wrong — 10 batches × 20 questions, targeting 50% accuracy."""
    batches = []
    for batch_id in range(10):
        batch_items = generate_items_with_opus(llm, """Generate {n} factual questions for a "predict your errors" benchmark.

CRITICAL: These questions must span a wide difficulty range. Approximately:
- 5 should be EASY (common knowledge, ~95% AI accuracy)
- 5 should be MEDIUM (requires expertise, ~70% AI accuracy)
- 5 should be HARD (specialist knowledge, ~40% AI accuracy)
- 5 should be VERY HARD (extremely obscure, ~15% AI accuracy)

This means a typical AI should get about 10-12 right and 8-10 wrong out of 20.

Each item must have:
- "question_id": integer 1-20
- "question": factual question
- "correct_answer": precise answer
- "accept_aliases": pipe-separated alternatives (or "")

HARD and VERY HARD examples:
- Exact population of a specific small city in a specific year
- Winners of minor academic awards
- Specific enzyme names in biochemical pathways
- Exact dates of minor treaties or laws
- Statistics from specific years in niche fields

Return a JSON array of 20 items.""", 20)

        # Package as batch
        questions = {}
        answers = {}
        aliases = {}
        for item in batch_items:
            qid = str(item.get("question_id", 0))
            questions[qid] = item.get("question", "")
            answers[qid] = item.get("correct_answer", "")
            aliases[qid] = item.get("accept_aliases", "")

        batches.append({
            "batch_id": batch_id,
            "questions_json": json.dumps(questions, ensure_ascii=False),
            "answers_json": json.dumps(answers, ensure_ascii=False),
            "aliases_json": json.dumps(aliases, ensure_ascii=False),
        })
    return batches


def gen_t10(llm):
    """T10: Difficulty Ranking — 5 sets × 30 questions, targeting 50% accuracy."""
    sets = []
    for set_id in range(5):
        set_items = generate_items_with_opus(llm, """Generate {n} factual questions for a difficulty ranking benchmark.

CRITICAL REQUIREMENT: These 30 questions must span an EXTREME difficulty range:
- 6 TRIVIAL (capitals, basic math, famous facts — ~99% AI accuracy)
- 6 EASY (general knowledge — ~85% AI accuracy)
- 6 MEDIUM (requires expertise — ~60% AI accuracy)
- 6 HARD (specialist knowledge — ~30% AI accuracy)
- 6 EXTREME (nearly impossible for AI — ~10% AI accuracy)

Each item must have:
- "question_id": integer 1-30
- "question": factual question
- "correct_answer": precise answer
- "accept_aliases": pipe-separated alternatives (or "")

Return a JSON array of 30 items.""", 30)

        questions = {}
        answers = {}
        aliases = {}
        for item in set_items:
            qid = str(item.get("question_id", 0))
            questions[qid] = item.get("question", "")
            answers[qid] = item.get("correct_answer", "")
            aliases[qid] = item.get("accept_aliases", "")

        sets.append({
            "set_id": set_id,
            "questions_json": json.dumps(questions, ensure_ascii=False),
            "answers_json": json.dumps(answers, ensure_ascii=False),
            "aliases_json": json.dumps(aliases, ensure_ascii=False),
        })
    return sets


def gen_t11(llm):
    """T11: Should I Attempt — 200 items targeting 50% accuracy."""
    items = []
    for diff, n in [("easy", 66), ("medium", 67), ("hard", 67)]:
        desc = {
            "easy": "common knowledge (~95% AI accuracy). Answering is always worthwhile.",
            "medium": "requires expertise (~50-70% AI accuracy). The skip decision matters.",
            "hard": "very obscure facts (~20-30% AI accuracy). Skipping is often optimal. Include deceptive items that APPEAR easy but have commonly wrong answers.",
        }[diff]
        batch = generate_items_with_opus(llm, f"""Generate {{n}} factual questions for an economic decision benchmark.
Difficulty: {diff} — {desc}

Rules of the game: +10 for correct, -15 for wrong, +1 for skip.
The AI must decide whether to answer or skip each question.

Each item must have:
- "question": factual question
- "correct_answer": precise answer
- "difficulty": "{diff}"
- "accept_aliases": pipe-separated alternatives (or "")

For HARD items, include:
- Deceptive questions where the obvious answer is wrong
- Questions requiring exact numerical values
- Questions about minor/obscure facts

Return a JSON array.""", n)
        for item in batch:
            item["difficulty"] = diff
        items.extend(batch)
    return items


def gen_t48(llm):
    """T48: Abstention ROC — 500 items across domains and difficulties."""
    items = []
    domains = ["science", "history", "geography", "mathematics", "literature",
               "technology", "sports", "politics", "economics", "medicine"]
    per_domain = 50
    for domain in domains:
        batch = generate_items_with_opus(llm, f"""Generate {{n}} factual questions about {domain} for a comprehensive metacognition benchmark.

Difficulty distribution:
- 10 EASY (common knowledge, ~95% AI accuracy)
- 15 MEDIUM (general expertise, ~75% AI accuracy)
- 15 HARD (specialist, ~40% AI accuracy)
- 10 VERY HARD (extremely obscure, ~15% AI accuracy)

Each item must have:
- "question": factual question
- "correct_answer": precise answer
- "domain": "{domain}"
- "difficulty": "easy"/"medium"/"hard"/"very_hard"
- "accept_aliases": pipe-separated alternatives (or "")

For HARD and VERY HARD, use genuinely obscure facts: specific statistics, minor figures,
exact dates, regional facts, technical specifications.

Return a JSON array.""", per_domain)
        for item in batch:
            item["domain"] = domain
        items.extend(batch)
    return items


def gen_simple_qa(llm, benchmark_id, n, extra_prompt=""):
    """Generic generator for benchmarks that just need hard factual questions."""
    items = generate_items_with_opus(llm, f"""Generate {{n}} factual questions for a metacognition benchmark.

Mix difficulty: ~30% easy, ~30% medium, ~40% hard/very hard.
Target: an AI model should get about 50-60% correct.

Each item must have:
- "question": factual question
- "correct_answer": precise answer
- "difficulty": "easy"/"medium"/"hard"
- "accept_aliases": pipe-separated alternatives (or "")

{extra_prompt}

Return a JSON array.""", n)
    return items


def gen_t17(llm):
    """T17: Self-Review Pipeline — 10 batches × 10 hard questions targeting 30-40% error rate."""
    batches = []
    for batch_id in range(10):
        batch_items = generate_items_with_opus(llm, """Generate {n} factual questions designed so that an AI model will get about 3-4 of them WRONG.

Mix: 4 medium questions (~80% AI accuracy), 3 hard (~40% accuracy), 3 very hard (~15% accuracy).

Each item must have:
- "question_id": integer 1-10
- "question": factual question
- "correct_answer": precise answer
- "accept_aliases": pipe-separated alternatives (or "")

Return a JSON array.""", 10)
        questions, answers, aliases = {}, {}, {}
        for item in batch_items:
            qid = str(item.get("question_id", 0))
            questions[qid] = item.get("question", "")
            answers[qid] = item.get("correct_answer", "")
            aliases[qid] = item.get("accept_aliases", "")
        batches.append({
            "batch_id": batch_id,
            "batch_questions": json.dumps(questions, ensure_ascii=False),
            "batch_answers": json.dumps(answers, ensure_ascii=False),
            "batch_aliases": json.dumps(aliases, ensure_ascii=False),
        })
    return batches


def gen_t18(llm):
    """T18: Planted Error Detection — 50 passages with planted errors + 15 error-free controls."""
    items = generate_items_with_opus(llm, """Generate {n} short factual passages (100-200 words each) for an error detection benchmark.

For EACH passage:
- Write a factual paragraph about a topic (science, history, geography, technology, etc.)
- Plant 1-2 SUBTLE factual errors (wrong dates, incorrect attributions, swapped numbers, wrong names)
- The errors should be plausible — not obviously absurd

Each item must have:
- "passage_id": integer
- "passage": the text with planted errors
- "errors": JSON array of objects with "position" (which sentence), "wrong" (the error), "correct" (what it should be)
- "num_errors": count of errors
- "topic": subject area

Return a JSON array.""", 50)
    # Add error-free controls
    controls = generate_items_with_opus(llm, """Generate {n} short factual passages (100-200 words each) that are COMPLETELY ACCURATE — no errors at all.

Each item must have:
- "passage_id": integer (starting from 100)
- "passage": factually correct text
- "errors": [] (empty array)
- "num_errors": 0
- "topic": subject area

Return a JSON array.""", 15)
    for c in controls:
        c["passage_id"] = c.get("passage_id", 0) + 100
    return items + controls


def gen_t19(llm):
    """T19: Math Verification Asymmetry — 60 solutions (30 correct + 30 with subtle errors) × 2 attributions."""
    solutions = generate_items_with_opus(llm, """Generate {n} math problem solutions for a verification asymmetry benchmark.

Generate 15 CORRECT solutions and 15 solutions with SUBTLE errors.
Errors should be hard to spot: sign mistakes, off-by-one, wrong constants, dropped terms.

Each item must have:
- "solution_id": integer
- "problem": the math problem statement
- "solution": the solution work (showing steps)
- "is_correct": true/false
- "error_description": if incorrect, describe the error (else "")

Cover: algebra, calculus, combinatorics, probability, geometry.

Return a JSON array.""", 30)
    # Duplicate each for self/student attribution
    items = []
    for sol in solutions:
        for attr in ["self", "student"]:
            items.append({**sol, "attribution": attr})
    return items


def gen_t26(llm):
    """T26: Iterative Self-Correction — 60 questions targeting 40-50% initial accuracy."""
    return generate_items_with_opus(llm, """Generate {n} factual and reasoning questions where an AI model will likely get about HALF of them wrong on the first attempt.

Mix of:
- 15 multi-step math problems with common pitfalls
- 15 tricky logic/reasoning questions
- 15 obscure factual questions
- 15 questions with commonly confused answers

Each item must have:
- "question": the question
- "correct_answer": precise answer
- "question_type": "math"/"logic"/"trivia"/"science"
- "accept_aliases": pipe-separated alternatives (or "")

Return a JSON array.""", 60)


def gen_t27(llm):
    """T27: Known/Unknown Sorting — 150 statements (50 true, 50 false, 50 ambiguous)."""
    items = []
    for category, n, desc in [
        ("clear_true", 50, "clearly true factual statements that are well-established"),
        ("clear_false", 50, "plausible-sounding but FALSE statements. Make them tricky — partially true, wrong by one detail, or common misconceptions"),
        ("ambiguous", 50, "genuinely AMBIGUOUS statements: partially true, depend on interpretation, have contested answers, or are true with important caveats. These should make even an expert pause."),
    ]:
        batch = generate_items_with_opus(llm, f"""Generate {{n}} {desc} for a knowledge boundary benchmark.

Each item must have:
- "statement": the factual statement
- "is_true": true/false/null (null for ambiguous)
- "difficulty": "easy"/"medium"/"hard"
- "domain": knowledge area

Return a JSON array.""", n)
        for item in batch:
            item["_category"] = category
        items.extend(batch)
    return items


def gen_t28(llm):
    """T28: Fabrication Detection Self-Test — 25 topics at knowledge boundary."""
    return generate_items_with_opus(llm, """Generate {n} topic descriptions for a fabrication detection benchmark.

Each topic should be something an AI model PARTIALLY knows about — not completely well-known, not completely obscure. The model should be able to state some true facts but likely fabricate others.

Good topics: minor historical events, obscure scientific phenomena, regional cultural practices, niche technology standards, lesser-known biological species.

Each item must have:
- "topic_id": integer
- "topic": topic description (e.g., "The ecology of the Socotra archipelago")
- "num_facts": 5 (always 5)
- "verification_sources": "[]" (placeholder)

Return a JSON array.""", 25)


def gen_t29(llm):
    """T29: Wikipedia Gap Test — 30 real obscure + 30 fabricated (realistic names)."""
    real = generate_items_with_opus(llm, """Generate {n} REAL but OBSCURE entities with questions about them.

These must be real, verifiable entities that are genuinely obscure — not famous.
Categories: scientists, historical figures, companies, places, scientific concepts.

Each item must have:
- "entity_name": the real entity name
- "entity_type": "real_obscure"
- "question": a factual question about this entity
- "correct_answer": the actual correct answer
- "domain": category

Return a JSON array.""", 30)

    fake = generate_items_with_opus(llm, """Generate {n} FABRICATED entities with REALISTIC names for a hallucination detection benchmark.

CRITICAL: Names must be VERY realistic — follow real naming conventions:
- People: common first+last names from appropriate cultures (e.g., "Henrik Lindqvist", "Maria Costanzo")
- Companies: "[Real surname] [Real industry word]" (e.g., "Barrett Semiconductors", "Nakamura Biotech")
- Places: plausible geographic names (e.g., "Porto Ferreira", "Khadipur")
- Concepts: "[Real scientist surname]-[Real term]" (e.g., "The Petersen-Holt conjecture")

DO NOT use obviously fake names like "Nexalith" or "Veridian."

Each item must have:
- "entity_name": the fabricated entity name
- "entity_type": "fabricated"
- "question": a question about it (answer should be "this entity does not exist")
- "correct_answer": "This entity does not exist"
- "domain": category

Return a JSON array.""", 30)
    return real + fake


def gen_t34(llm):
    """T34: Synthetic Entity Recognition — 25 real obscure + 25 realistic synthetic."""
    real = generate_items_with_opus(llm, """Generate {n} REAL but very OBSCURE entities across 5 categories.

5 each of: person, company, city, compound, concept.
These must be real and verifiable but genuinely obscure.

Each item must have:
- "entity_name": real name
- "entity_type": "real"
- "entity_category": "person"/"company"/"city"/"compound"/"concept"
- "question": factual question about it
- "correct_answer": correct answer

Return a JSON array.""", 25)

    fake = generate_items_with_opus(llm, """Generate {n} FABRICATED entities with EXTREMELY realistic names across 5 categories.

5 each of: person, company, city, compound, concept.

CRITICAL naming rules:
- Person: use common names from relevant cultures (e.g., "Dr. James Whitfield", "Prof. Yuki Tanaka")
- Company: "[Surname] [Industry]" pattern (e.g., "Morrison Dynamics", "Chen Materials")
- City: plausible place names (e.g., "San Marcos del Norte", "Altenburg-Ost")
- Compound: real chemistry naming patterns (e.g., "3-methylpentyl acetate", "diethyl-4-aminobenzoate")
- Concept: "[Scientist]-[Term]" (e.g., "The Brouwer-Kolmogorov theorem", "Paulson equilibrium")

NONE of these should actually exist.

Each item must have:
- "entity_name": fabricated name
- "entity_type": "synthetic"
- "entity_category": category
- "question": question about it
- "correct_answer": "entity_does_not_exist"

Return a JSON array.""", 25)
    return real + fake


def gen_t38(llm):
    """T38: Help-Seeking Behavior — 20 complete + 20 incomplete + 10 borderline."""
    complete = generate_items_with_opus(llm, """Generate {n} questions that have ALL necessary information to answer.

Mix of difficulties. Include both simple and complex questions.

Each item must have:
- "question": complete question
- "is_complete": true
- "missing_info": ""
- "correct_answer": the answer
- "question_type": "math"/"factual"/"contextual"

Return a JSON array.""", 20)

    incomplete = generate_items_with_opus(llm, """Generate {n} questions that are MISSING critical information needed to answer.

Make the missing info SUBTLE — not obviously absent. Examples:
- "What is the monthly payment?" (missing: loan amount, term, rate)
- "How long will the trip take?" (missing: mode of transport, route)
- "What is the profit margin?" (missing: which company, which year)

Each item must have:
- "question": the incomplete question
- "is_complete": false
- "missing_info": what's missing
- "correct_answer": "Need more information"
- "question_type": "math"/"factual"/"contextual"

Return a JSON array.""", 20)

    borderline = generate_items_with_opus(llm, """Generate {n} questions that are BORDERLINE — technically answerable with reasonable assumptions, but asking for clarification would be better.

Examples: "What is the best programming language?" (subjective), "How many calories in a sandwich?" (depends on sandwich).

Each item must have:
- "question": the borderline question
- "is_complete": false
- "missing_info": what assumptions would need to be made
- "correct_answer": "Depends on context"
- "question_type": "contextual"

Return a JSON array.""", 10)
    return complete + incomplete + borderline


def gen_t43(llm):
    """T43: Delegation Judgment — 60 tasks including 20 borderline."""
    clear = generate_items_with_opus(llm, """Generate {n} task descriptions for a delegation judgment benchmark.

10 clearly for AI, 10 clearly for human, 10 clearly for tools, 10 for AI-with-caveat.

Each item must have:
- "task_description": what needs to be done
- "ideal_handler": "ai"/"human"/"tool"/"ai_with_caveat"
- "reason": why this handler
- "category": task category

Return a JSON array.""", 40)

    borderline = generate_items_with_opus(llm, """Generate {n} BORDERLINE task descriptions where the delegation decision is genuinely debatable.

Examples:
- "Review this contract for common red flags" (AI with caveat? Human?)
- "Draft an initial response to this customer complaint" (AI? Human?)
- "Triage emergency room patients" (Tool? Human?)

Each item must have:
- "task_description": the task
- "ideal_handler": your best judgment of the right handler
- "reason": why (including the debate)
- "category": task category

Return a JSON array.""", 20)
    return clear + borderline


# Map benchmark IDs to generators
GENERATORS = {
    "t01": gen_t01,
    "t02": gen_t02,
    "t03": gen_t03,
    "t04": gen_t04,
    "t06": gen_t06,
    "t07": gen_t07,
    "t09": gen_t09,
    "t10": gen_t10,
    "t11": gen_t11,
    "t17": gen_t17,
    "t18": gen_t18,
    "t19": gen_t19,
    "t26": gen_t26,
    "t27": gen_t27,
    "t28": gen_t28,
    "t29": gen_t29,
    "t34": gen_t34,
    "t38": gen_t38,
    "t43": gen_t43,
    "t48": gen_t48,
}


def main():
    parser = argparse.ArgumentParser(description="Generate v2 datasets using Opus 4.6")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--output-dir", default="benchmarks/datasets")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of benchmarks to generate in parallel")
    args = parser.parse_args()

    target_ids = args.benchmarks or list(GENERATORS.keys())

    # Generate benchmarks in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _gen_one(bid):
        if bid not in GENERATORS:
            logger.warning("No Opus generator for %s", bid)
            return bid, 0, None
        try:
            llm = get_opus_client(args.region)
            items = GENERATORS[bid](llm)
            save_dataset(items, bid, args.output_dir)
            logger.info("[%s] DONE — %d items", bid, len(items))
            return bid, len(items), None
        except Exception as e:
            logger.error("[%s] FAILED: %s", bid, e)
            return bid, 0, str(e)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(_gen_one, bid): bid for bid in target_ids}
        for future in as_completed(futures):
            bid, n, err = future.result()
            if err:
                print(f"  [{bid}] ERROR: {err}")
            else:
                print(f"  [{bid}] {n} items")

    logger.info("Done. Datasets saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
