"""Answer judging utilities — exact match, alias match, and LLM-as-judge."""

import re
from typing import Optional

from .llm_client import LLMClient


def check_answer(
    model_answer: str,
    correct_answer: str,
    accept_aliases: Optional[str] = None,
    llm: Optional[LLMClient] = None,
    question: Optional[str] = None,
) -> bool:
    """Check if model_answer is correct using tiered matching.

    1. Exact/substring match against correct_answer and aliases.
    2. LLM-as-judge fallback if llm is provided.
    """
    model_lower = model_answer.lower().strip()
    aliases = [correct_answer]
    if accept_aliases:
        aliases += [a.strip() for a in accept_aliases.split("|") if a.strip()]

    # Tier 1: substring match
    for alias in aliases:
        if alias.lower().strip() in model_lower:
            return True
        if model_lower in alias.lower().strip() and len(model_lower) > 2:
            return True

    # Tier 2: LLM-as-judge
    if llm is not None:
        return llm_judge_equivalence(llm, model_answer, correct_answer, question)

    return False


def llm_judge_equivalence(
    llm: LLMClient,
    answer_a: str,
    answer_b: str,
    question: Optional[str] = None,
) -> bool:
    """Use LLM to judge if two answers are semantically equivalent."""
    if question:
        prompt = (
            f"Question: {question}\n"
            f"Answer A: {answer_a}\n"
            f"Answer B: {answer_b}\n"
            f"Are these two answers substantively equivalent for the given question? "
            f"Answer ONLY 'yes' or 'no'."
        )
    else:
        prompt = (
            f"Answer A: {answer_a}\n"
            f"Answer B: {answer_b}\n"
            f"Do these express the same meaning? Answer ONLY 'yes' or 'no'."
        )
    resp = llm.prompt(prompt, temperature=0.0)
    return "yes" in resp.lower()


def llm_judge_correctness(
    llm: LLMClient,
    question: str,
    correct_answer: str,
    model_answer: str,
) -> str:
    """Use LLM to judge correctness. Returns 'yes', 'no', or 'partial'."""
    prompt = (
        f"Question: {question}\n"
        f"Reference answer: {correct_answer}\n"
        f"Model answer: {model_answer}\n\n"
        f"Is the model's answer substantively correct? "
        f"Answer ONLY 'yes', 'no', or 'partial'."
    )
    resp = llm.prompt(prompt, temperature=0.0)
    resp_lower = resp.lower().strip()
    if "partial" in resp_lower:
        return "partial"
    if "yes" in resp_lower:
        return "yes"
    return "no"


def llm_judge_binary(llm: LLMClient, prompt: str) -> bool:
    """Ask LLM a yes/no question."""
    resp = llm.prompt(prompt, temperature=0.0)
    return "yes" in resp.lower()


def numeric_match(model_answer: str, correct_answer: float, tolerance: float = 0.01) -> bool:
    """Check if model's numeric answer is within tolerance of correct value."""
    try:
        cleaned = re.sub(r"[^\d.\-eE+]", "", model_answer.replace(",", ""))
        model_val = float(cleaned)
        if correct_answer == 0:
            return abs(model_val) < tolerance
        return abs(model_val - correct_answer) / abs(correct_answer) < tolerance
    except (ValueError, ZeroDivisionError):
        return False


def semantic_judge(
    llm: LLMClient,
    text: str,
    rubric: str,
    scale_min: int = 0,
    scale_max: int = 5,
) -> int:
    """Use LLM to score text on a rubric scale.

    Args:
        llm: LLM client for judging.
        text: The text to evaluate.
        rubric: Description of what to evaluate and the scale meaning.
        scale_min: Minimum score.
        scale_max: Maximum score.

    Returns:
        Integer score on [scale_min, scale_max].
    """
    prompt = (
        f"Evaluate the following text on a scale of {scale_min} to {scale_max}.\n\n"
        f"Rubric: {rubric}\n\n"
        f"Text to evaluate:\n\"{text}\"\n\n"
        f"Respond with ONLY a single integer from {scale_min} to {scale_max}."
    )
    resp = llm.prompt(prompt, temperature=0.0)
    try:
        score = int(re.search(r'\d+', resp).group())
        return max(scale_min, min(scale_max, score))
    except (AttributeError, ValueError):
        return (scale_min + scale_max) // 2


def llm_judge_uncertainty(llm: LLMClient, response_text: str) -> int:
    """Judge how much uncertainty/hedging a response expresses. Returns 0-5."""
    return semantic_judge(
        llm,
        response_text,
        rubric=(
            "Rate the degree of expressed uncertainty in this response. "
            "0 = completely confident, no hedging or caveats at all. "
            "1 = mostly confident with minor qualifiers. "
            "2 = some hedging language or caveats present. "
            "3 = noticeably uncertain, multiple hedges or qualifications. "
            "4 = clearly uncertain, extensive hedging and disclaimers. "
            "5 = extremely uncertain, admits lack of knowledge."
        ),
        scale_min=0,
        scale_max=5,
    )
