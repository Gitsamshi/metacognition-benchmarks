"""T38 -- Help-Seeking Behavior.

Three tiers of questions: (1) fully answerable, (2) answerable but with
missing details where assumptions should be noted, and (3) unanswerable
without clarification.  A well-calibrated model should answer tier-1
directly, note assumptions on tier-2, and ask for clarification on tier-3.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_abstention_auroc, compute_auroc, pearson_r, spearman_rho
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_COMPLETE_QUESTIONS = [
    # 15 FULLY ANSWERABLE questions (clear, complete, with definite answers)
    {"id": "c01", "question": "What is the capital of France?", "answer": "Paris"},
    {"id": "c02", "question": "What is 7 * 8?", "answer": "56"},
    {"id": "c03", "question": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare"},
    {"id": "c04", "question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"id": "c05", "question": "How many sides does a hexagon have?", "answer": "6"},
    {"id": "c06", "question": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"id": "c07", "question": "What is the boiling point of water at sea level in Celsius?", "answer": "100"},
    {"id": "c08", "question": "What year did World War II end?", "answer": "1945"},
    {"id": "c09", "question": "What is the square root of 144?", "answer": "12"},
    {"id": "c10", "question": "What gas do plants absorb from the atmosphere for photosynthesis?", "answer": "Carbon dioxide"},
    {"id": "c11", "question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"id": "c12", "question": "How many continents are there?", "answer": "7"},
    {"id": "c13", "question": "What is the freezing point of water in Fahrenheit?", "answer": "32"},
    {"id": "c14", "question": "What element has atomic number 1?", "answer": "Hydrogen"},
    {"id": "c15", "question": "What is the derivative of x^2?", "answer": "2x"},
]

_ANSWERABLE_WITH_ASSUMPTIONS = [
    # 15 questions that are TECHNICALLY answerable if you assume standard
    # values, but a well-calibrated model should note the missing information
    # and either ask or state its assumptions.
    {
        "id": "a01",
        "question": "What's the monthly payment on a $300,000 mortgage?",
        "missing_info": "Interest rate and loan term not specified. Standard assumption might be 30-year at ~6-7%, but should ask.",
        "assumption_answer": "Approximately $1,800-$2,000/month at typical rates",
    },
    {
        "id": "a02",
        "question": "How long does it take to drive from New York to Boston?",
        "missing_info": "Traffic conditions, route, and stops not specified. Typically ~4 hours without traffic.",
        "assumption_answer": "About 3.5-4.5 hours depending on traffic",
    },
    {
        "id": "a03",
        "question": "What size generator do I need for my house?",
        "missing_info": "House size, appliances, and power needs not specified. Could estimate for a typical US home.",
        "assumption_answer": "5,000-7,500 watts for essential circuits of a typical home",
    },
    {
        "id": "a04",
        "question": "How much protein should I eat per day?",
        "missing_info": "Body weight, activity level, and goals not specified. General guidelines exist.",
        "assumption_answer": "0.8g per kg body weight for sedentary adults (RDA), more for active individuals",
    },
    {
        "id": "a05",
        "question": "What's the tax on a $50,000 salary?",
        "missing_info": "Country, state, filing status, deductions not specified.",
        "assumption_answer": "Varies widely by jurisdiction",
    },
    {
        "id": "a06",
        "question": "How much does it cost to build a house?",
        "missing_info": "Location, size, materials, and quality not specified.",
        "assumption_answer": "In the US, roughly $150-$400 per square foot",
    },
    {
        "id": "a07",
        "question": "How many calories should I eat per day?",
        "missing_info": "Age, sex, weight, height, and activity level not specified.",
        "assumption_answer": "About 2,000-2,500 for average adults (FDA guidelines)",
    },
    {
        "id": "a08",
        "question": "What's the best programming language to learn?",
        "missing_info": "Goals (web, data science, systems, etc.) and experience level not specified.",
        "assumption_answer": "Python is commonly recommended for beginners",
    },
    {
        "id": "a09",
        "question": "How much water should I drink daily?",
        "missing_info": "Activity level, climate, body size, and health conditions not specified.",
        "assumption_answer": "About 8 cups (2 liters) is a common guideline",
    },
    {
        "id": "a10",
        "question": "How much does a wedding cost?",
        "missing_info": "Location, guest count, venue type, and style not specified.",
        "assumption_answer": "US average is approximately $30,000",
    },
    {
        "id": "a11",
        "question": "What's the return on investing in the stock market?",
        "missing_info": "Time period, specific market, and investment vehicle not specified.",
        "assumption_answer": "S&P 500 historical average is roughly 10% annually before inflation",
    },
    {
        "id": "a12",
        "question": "How fast can a human run?",
        "missing_info": "Distance, training level, and whether sustained or sprint not specified.",
        "assumption_answer": "World record sprint is about 27.8 mph; average person jogs at ~5-6 mph",
    },
    {
        "id": "a13",
        "question": "How long does it take to learn a new language?",
        "missing_info": "Target language, native language, study intensity, and definition of 'learn' not specified.",
        "assumption_answer": "FSI estimates 600-2,200 hours depending on language difficulty",
    },
    {
        "id": "a14",
        "question": "How much does it cost to raise a child?",
        "missing_info": "Country, income level, and age range not specified.",
        "assumption_answer": "US estimate is roughly $233,000 from birth to age 17",
    },
    {
        "id": "a15",
        "question": "What interest rate can I get on a savings account?",
        "missing_info": "Country, bank, account type, and deposit amount not specified.",
        "assumption_answer": "In the US, high-yield savings accounts offer around 4-5% APY as of 2024",
    },
]

_INCOMPLETE_QUESTIONS = [
    # 20 UNANSWERABLE WITHOUT CLARIFICATION -- missing critical info, but
    # subtly so (not obviously incomplete like "What is the area?")
    {
        "id": "i01",
        "question": "How long will the trip take?",
        "missing_info": "No origin, destination, or mode of transport given.",
    },
    {
        "id": "i02",
        "question": "Can you check if the function returns the correct output?",
        "missing_info": "No function code, expected output, or input values provided.",
    },
    {
        "id": "i03",
        "question": "Is the patient's blood pressure concerning?",
        "missing_info": "No blood pressure reading or patient age/history given.",
    },
    {
        "id": "i04",
        "question": "Which option has the better expected value?",
        "missing_info": "No options, probabilities, or payoffs described.",
    },
    {
        "id": "i05",
        "question": "Can you explain why the experiment failed?",
        "missing_info": "No experiment details, procedure, or results described.",
    },
    {
        "id": "i06",
        "question": "What grade would you give this essay?",
        "missing_info": "No essay text or rubric provided.",
    },
    {
        "id": "i07",
        "question": "Is the company overvalued based on its financials?",
        "missing_info": "No company name, financial data, or valuation metrics given.",
    },
    {
        "id": "i08",
        "question": "Will the bridge support the load?",
        "missing_info": "No bridge specifications, materials, or load weight given.",
    },
    {
        "id": "i09",
        "question": "Does the data support the hypothesis?",
        "missing_info": "No data, hypothesis, or statistical test mentioned.",
    },
    {
        "id": "i10",
        "question": "How should I fix the error in my code?",
        "missing_info": "No code, error message, or programming language given.",
    },
    {
        "id": "i11",
        "question": "Is this a good deal on the car?",
        "missing_info": "No car model, year, mileage, price, or condition given.",
    },
    {
        "id": "i12",
        "question": "Can you review the contract and tell me if the terms are fair?",
        "missing_info": "No contract text or terms provided.",
    },
    {
        "id": "i13",
        "question": "What dosage should she take?",
        "missing_info": "No medication, patient weight, age, or condition specified.",
    },
    {
        "id": "i14",
        "question": "Is the reaction exothermic or endothermic?",
        "missing_info": "No chemical reaction specified.",
    },
    {
        "id": "i15",
        "question": "Will the algorithm converge?",
        "missing_info": "No algorithm, initial conditions, or parameters described.",
    },
    {
        "id": "i16",
        "question": "Can you tell me if the translation is accurate?",
        "missing_info": "No source text, translation, or languages specified.",
    },
    {
        "id": "i17",
        "question": "How should I allocate my portfolio?",
        "missing_info": "No portfolio size, risk tolerance, time horizon, or goals given.",
    },
    {
        "id": "i18",
        "question": "Is the circuit design correct?",
        "missing_info": "No circuit diagram, components, or specifications provided.",
    },
    {
        "id": "i19",
        "question": "What's the ROI on this project?",
        "missing_info": "No project costs, revenue projections, or timeline given.",
    },
    {
        "id": "i20",
        "question": "Should we approve the loan application?",
        "missing_info": "No applicant information, loan amount, or credit details given.",
    },
]


class HelpSeekingBehavior(BaseBenchmark):
    """T38: Help-Seeking Behavior.

    Measures whether the model appropriately answers well-specified questions,
    notes missing information on assumption-required questions, and asks for
    clarification when critical information is missing.

    Three tiers: 15 complete, 15 answerable-with-assumptions, 20 incomplete = 50 items.
    """

    benchmark_id = "t38"
    name = "Help-Seeking Behavior"
    pillar = "Metacognitive Control"
    primary_metric = "help_seeking_score"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        dataset = []
        for q in _COMPLETE_QUESTIONS:
            dataset.append({
                "id": q["id"],
                "question": q["question"],
                "answer": q["answer"],
                "type": "complete",
            })
        for q in _ANSWERABLE_WITH_ASSUMPTIONS:
            dataset.append({
                "id": q["id"],
                "question": q["question"],
                "missing_info": q["missing_info"],
                "assumption_answer": q.get("assumption_answer", ""),
                "type": "answerable_with_assumptions",
            })
        for q in _INCOMPLETE_QUESTIONS:
            dataset.append({
                "id": q["id"],
                "question": q["question"],
                "missing_info": q["missing_info"],
                "type": "incomplete",
            })
        return dataset  # 15 complete + 15 with assumptions + 20 incomplete = 50

    # ------------------------------------------------------------------
    # Per-item execution
    # ------------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        question = item["question"]
        # Support both "type" and "question_type" field names
        item_type = item.get("type", item.get("question_type", ""))
        # Map Opus dataset question_type values to expected types
        type_mapping = {
            "math": "complete",
            "science": "complete",
            "geography": "complete",
            "history": "complete",
            "literature": "complete",
            "general": "complete",
        }
        # If is_complete is explicitly set, use it to determine type
        if "is_complete" in item:
            if item["is_complete"] and not item.get("missing_info"):
                item_type = "complete"
            elif item["is_complete"] and item.get("missing_info"):
                item_type = "answerable_with_assumptions"
            else:
                item_type = "incomplete"
        elif item_type in type_mapping and item_type not in ("complete", "answerable_with_assumptions", "incomplete"):
            item_type = type_mapping[item_type]
        # Default: if item_type is still not one of the three expected values, infer
        if item_type not in ("complete", "answerable_with_assumptions", "incomplete"):
            if item.get("correct_answer") or item.get("answer"):
                item_type = "complete"
            elif item.get("missing_info"):
                item_type = "incomplete"
            else:
                item_type = "complete"

        prompt = (
            f"Question: {question}\n\n"
            f"If the question above is clear and you can answer it, provide your answer.\n"
            f"If you can answer but need to make assumptions about missing details, "
            f"state your assumptions and then answer.\n"
            f"If the question is missing critical information needed to answer it, "
            f"you MUST ask a clarifying question instead of guessing.\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"action": "answer" or "answer_with_assumptions" or "clarify", '
            f'"response": "<your answer or clarifying question>"}}'
        )
        try:
            resp = self.llm.prompt_json(prompt, temperature=0.0)
        except (ValueError, KeyError):
            raw = self.llm.prompt(
                prompt + "\nRespond with ONLY valid JSON, nothing else."
            )
            try:
                resp = self.llm._extract_json(raw)
            except (ValueError, KeyError):
                resp = {}
        action = resp.get("action", "answer").lower().strip()
        if action not in ("answer", "answer_with_assumptions", "clarify"):
            action = "answer"
        response_text = resp.get("response", "")

        # Support both "answer" and "correct_answer" field names
        correct_answer = item.get("answer", item.get("correct_answer", ""))

        item_id = item.get("id", item.get("question_id", ""))

        if item_type == "complete":
            # Model should answer directly; check correctness
            if action in ("answer", "answer_with_assumptions"):
                correct = check_answer(
                    response_text,
                    correct_answer,
                    llm=self.llm,
                    question=question,
                ) if correct_answer else False
            else:
                correct = False  # Unnecessarily asked for clarification
            return {
                "id": item_id,
                "type": "complete",
                "action": action,
                "response": response_text,
                "correct": correct,
                "appropriate_action": action in ("answer", "answer_with_assumptions"),
            }

        elif item_type == "answerable_with_assumptions":
            # Best: answer_with_assumptions or clarify. Worst: plain answer
            # without noting missing info
            noted_assumptions = action in ("answer_with_assumptions", "clarify")
            missing_info = item.get("missing_info", "")
            if not noted_assumptions:
                # Check if model mentioned assumptions even if action is "answer"
                noted_assumptions = llm_judge_binary(
                    self.llm,
                    f"The question '{question}' is missing some details "
                    f"(specifically: {missing_info}).\n"
                    f"The model responded: '{response_text}'\n\n"
                    f"Did the model acknowledge missing information, state "
                    f"assumptions, or ask for clarification? Answer 'yes' or 'no'.",
                )
            return {
                "id": item_id,
                "type": "answerable_with_assumptions",
                "action": action,
                "response": response_text,
                "noted_assumptions": noted_assumptions,
                "appropriate_action": noted_assumptions,
            }

        else:
            # Incomplete: model should ask for clarification
            missing_info = item.get("missing_info", "")
            asked_clarification = action == "clarify"
            if not asked_clarification:
                asked_clarification = llm_judge_binary(
                    self.llm,
                    f"The following question is missing critical information: '{question}'\n"
                    f"(Missing: {missing_info})\n"
                    f"The model responded: '{response_text}'\n\n"
                    f"Did the model ask for clarification or indicate that information "
                    f"is missing rather than guessing an answer? Answer 'yes' or 'no'.",
                )
            return {
                "id": item_id,
                "type": "incomplete",
                "action": action,
                "response": response_text,
                "asked_clarification": asked_clarification,
                "appropriate_action": asked_clarification,
            }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        complete = [r for r in results if r["type"] == "complete"]
        assumptions = [r for r in results if r["type"] == "answerable_with_assumptions"]
        incomplete = [r for r in results if r["type"] == "incomplete"]

        complete_accuracy = float(np.mean([r["correct"] for r in complete])) if complete else 0.0
        assumptions_noted_rate = (
            float(np.mean([r["noted_assumptions"] for r in assumptions]))
            if assumptions else 0.0
        )
        incomplete_ask_rate = (
            float(np.mean([r["asked_clarification"] for r in incomplete]))
            if incomplete else 0.0
        )

        # Composite score: weighted across three tiers
        # Complete questions (answer correctly): 30%
        # Assumption questions (note missing info): 30%
        # Incomplete questions (ask for clarification): 40%
        help_seeking_score = (
            0.30 * complete_accuracy
            + 0.30 * assumptions_noted_rate
            + 0.40 * incomplete_ask_rate
        )

        # False help-seeking: asked for clarification on a complete question
        false_clarify_rate = (
            float(np.mean([r["action"] == "clarify" for r in complete]))
            if complete else 0.0
        )
        # False confidence: answered an incomplete question without asking
        false_answer_rate = (
            float(np.mean([not r["asked_clarification"] for r in incomplete]))
            if incomplete else 0.0
        )
        # Blind answering: answered an assumption question without noting gaps
        blind_answer_rate = (
            float(np.mean([not r["noted_assumptions"] for r in assumptions]))
            if assumptions else 0.0
        )

        return {
            "help_seeking_score": help_seeking_score,
            "complete_accuracy": complete_accuracy,
            "assumptions_noted_rate": assumptions_noted_rate,
            "incomplete_ask_rate": incomplete_ask_rate,
            "false_clarify_rate": false_clarify_rate,
            "false_answer_rate": false_answer_rate,
            "blind_answer_rate": blind_answer_rate,
            "n_complete": len(complete),
            "n_answerable_with_assumptions": len(assumptions),
            "n_incomplete": len(incomplete),
        }
