"""T37 — Adaptive Strategy Selection.

The model is presented with math/logic problems alongside three candidate
solution strategies.  It must pick the strategy most likely to succeed, then
solve the problem using that strategy.  We also test the model on the other
two strategies to measure opportunity cost and strategy‑awareness.
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

_PROBLEMS = [
    # ---- arithmetic / algebra ----
    {
        "problem_id": "arith_01",
        "problem": "Compute 47 * 83.",
        "answer": "3901",
        "strategies": [
            {"id": "direct", "name": "Direct multiplication", "description": "Multiply the two numbers directly using the standard algorithm."},
            {"id": "decompose", "name": "Decomposition", "description": "Break one factor into parts: 47*83 = 47*80 + 47*3."},
            {"id": "estimate", "name": "Estimation & adjustment", "description": "Round to 50*80=4000, then subtract corrections."},
        ],
    },
    {
        "problem_id": "arith_02",
        "problem": "Compute 123 + 456 + 789.",
        "answer": "1368",
        "strategies": [
            {"id": "sequential", "name": "Sequential addition", "description": "Add the numbers one by one from left to right."},
            {"id": "grouping", "name": "Smart grouping", "description": "Group numbers to make round sums, e.g., 123+789=912, then +456."},
            {"id": "column", "name": "Column addition", "description": "Line up digits and add column by column with carries."},
        ],
    },
    {
        "problem_id": "arith_03",
        "problem": "What is 15% of 260?",
        "answer": "39",
        "strategies": [
            {"id": "decimal", "name": "Decimal conversion", "description": "Convert 15% to 0.15 and multiply: 0.15 * 260."},
            {"id": "split", "name": "Percentage splitting", "description": "10% of 260 = 26, 5% = 13, sum = 39."},
            {"id": "fraction", "name": "Fraction approach", "description": "15% = 3/20, so compute 260 * 3 / 20."},
        ],
    },
    {
        "problem_id": "arith_04",
        "problem": "Compute 2^10.",
        "answer": "1024",
        "strategies": [
            {"id": "repeated", "name": "Repeated multiplication", "description": "Multiply 2 by itself 10 times sequentially."},
            {"id": "doubling", "name": "Successive doubling", "description": "Start at 2, keep doubling: 2,4,8,16,32,64,128,256,512,1024."},
            {"id": "squaring", "name": "Exponentiation by squaring", "description": "2^10 = (2^5)^2 = 32^2 = 1024."},
        ],
    },
    {
        "problem_id": "arith_05",
        "problem": "Solve for x: 3x + 7 = 22.",
        "answer": "5",
        "strategies": [
            {"id": "algebraic", "name": "Algebraic manipulation", "description": "Subtract 7 from both sides then divide by 3."},
            {"id": "guess_check", "name": "Guess and check", "description": "Try integer values until the equation holds."},
            {"id": "reverse", "name": "Working backward", "description": "Start from 22, subtract 7 to get 15, divide by 3 to get 5."},
        ],
    },
    # ---- logic / reasoning ----
    {
        "problem_id": "logic_01",
        "problem": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        "answer": "No",
        "strategies": [
            {"id": "venn", "name": "Venn diagram", "description": "Draw overlapping circles and check set relationships."},
            {"id": "syllogism", "name": "Formal syllogism", "description": "Apply syllogistic rules to the premises."},
            {"id": "counterexample", "name": "Counterexample search", "description": "Try to construct a scenario where the conclusion is false while premises hold."},
        ],
    },
    {
        "problem_id": "logic_02",
        "problem": "If it rains, the ground is wet. The ground is wet. Did it rain?",
        "answer": "Not necessarily",
        "strategies": [
            {"id": "truth_table", "name": "Truth table", "description": "Enumerate all truth-value combinations for the conditional."},
            {"id": "fallacy", "name": "Fallacy identification", "description": "Recognize affirming the consequent as a logical fallacy."},
            {"id": "counterexample", "name": "Counterexample", "description": "Find an alternative cause for the ground being wet (sprinkler)."},
        ],
    },
    {
        "problem_id": "logic_03",
        "problem": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "answer": "$0.05",
        "strategies": [
            {"id": "algebra", "name": "Set up equations", "description": "Let ball = x, bat = x + 1.00, solve x + (x+1) = 1.10."},
            {"id": "intuition_check", "name": "Intuition + verification", "description": "Note the intuitive answer $0.10, verify it, adjust."},
            {"id": "guess_check", "name": "Guess and check", "description": "Try $0.05: ball=$0.05, bat=$1.05, total=$1.10. Check."},
        ],
    },
    {
        "problem_id": "logic_04",
        "problem": "Three people check into a hotel room that costs $30. They each pay $10. Later the manager realizes it should be $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. Each person paid $9 (total $27) plus the bellboy's $2 = $29. Where is the missing dollar?",
        "answer": "There is no missing dollar; the $27 already includes the bellboy's $2.",
        "strategies": [
            {"id": "trace", "name": "Trace every dollar", "description": "Track the flow of all 30 dollars from start to finish."},
            {"id": "reframe", "name": "Reframe the arithmetic", "description": "Show the question's addition is misleading — $27 paid minus $25 room minus $2 bellboy = $0."},
            {"id": "table", "name": "Balance table", "description": "Create a ledger of who holds what at each step."},
        ],
    },
    {
        "problem_id": "logic_05",
        "problem": "You have 8 identical-looking balls. One is heavier. Using a balance scale, what is the minimum number of weighings needed to find the heavy ball?",
        "answer": "2",
        "strategies": [
            {"id": "ternary", "name": "Ternary search", "description": "Divide into 3 groups of ~3, weigh 2 groups to narrow down."},
            {"id": "binary", "name": "Binary search", "description": "Split into halves and weigh, then continue halving."},
            {"id": "exhaustive", "name": "Enumerate cases", "description": "List all possible weighing outcomes and check coverage."},
        ],
    },
    # ---- word / combination ----
    {
        "problem_id": "combo_01",
        "problem": "How many ways can you arrange the letters in the word 'BOOK'?",
        "answer": "12",
        "strategies": [
            {"id": "formula", "name": "Permutation with repetition formula", "description": "4! / 2! = 24/2 = 12."},
            {"id": "enumerate", "name": "Systematic enumeration", "description": "List all distinct arrangements manually."},
            {"id": "slot", "name": "Slot method", "description": "Choose positions for repeated letters, then fill the rest."},
        ],
    },
    {
        "problem_id": "combo_02",
        "problem": "A pizza place offers 3 toppings. How many different pizzas with exactly 2 toppings can you make?",
        "answer": "3",
        "strategies": [
            {"id": "combination", "name": "Combination formula", "description": "C(3,2) = 3."},
            {"id": "list", "name": "Listing", "description": "Enumerate: AB, AC, BC."},
            {"id": "complement", "name": "Complement counting", "description": "Choose which topping to leave out: 3 ways."},
        ],
    },
    {
        "problem_id": "combo_03",
        "problem": "What is the sum of the first 10 positive integers?",
        "answer": "55",
        "strategies": [
            {"id": "gauss", "name": "Gauss formula", "description": "n(n+1)/2 = 10*11/2 = 55."},
            {"id": "sequential", "name": "Sequential addition", "description": "Add 1+2+3+...+10 step by step."},
            {"id": "pairing", "name": "Pairing method", "description": "Pair 1+10, 2+9, ... each sums to 11; 5 pairs = 55."},
        ],
    },
    {
        "problem_id": "combo_04",
        "problem": "Find the GCD of 48 and 18.",
        "answer": "6",
        "strategies": [
            {"id": "euclidean", "name": "Euclidean algorithm", "description": "48 = 2*18 + 12; 18 = 1*12 + 6; 12 = 2*6 + 0; GCD=6."},
            {"id": "factorization", "name": "Prime factorization", "description": "48=2^4*3, 18=2*3^2; common = 2*3 = 6."},
            {"id": "subtraction", "name": "Repeated subtraction", "description": "Subtract the smaller from the larger until equal."},
        ],
    },
    {
        "problem_id": "combo_05",
        "problem": "Convert the binary number 11010 to decimal.",
        "answer": "26",
        "strategies": [
            {"id": "positional", "name": "Positional notation", "description": "1*16 + 1*8 + 0*4 + 1*2 + 0*1 = 26."},
            {"id": "doubling", "name": "Horner's method / doubling", "description": "Start from left: 1, 1*2+1=3, 3*2+0=6, 6*2+1=13, 13*2+0=26."},
            {"id": "table", "name": "Powers of 2 table", "description": "Write out powers of 2 under each bit and sum those with 1s."},
        ],
    },
]


class AdaptiveStrategySelection(BaseBenchmark):
    """T37: Adaptive Strategy Selection.

    Present the model with math/logic problems and three solution strategies.
    The model picks one strategy, solves the problem with it, and is also
    tested with the other two strategies.  The headline metric is the fraction
    of problems where the chosen strategy led to a correct answer.
    """

    benchmark_id = "t37"
    name = "Adaptive Strategy Selection"
    pillar = "Metacognitive Control"
    primary_metric = "strategy_hit_rate"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        return list(_PROBLEMS)

    # ------------------------------------------------------------------
    # Per‑item execution
    # ------------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        problem = item["problem"]
        strategies = item["strategies"]
        correct_answer = item["answer"]

        strategy_block = "\n".join(
            f"  {i+1}. [{s['id']}] {s['name']} — {s['description']}"
            for i, s in enumerate(strategies)
        )

        # Step 1 — ask the model to choose a strategy
        choose_prompt = (
            f"Problem: {problem}\n\n"
            f"Available strategies:\n{strategy_block}\n\n"
            f"Which strategy would you choose to solve this problem? "
            f"Respond with ONLY valid JSON: "
            f'{{"chosen_strategy_id": "<id>", "reason": "<brief reason>"}}'
        )
        choice = self.llm.prompt_json(choose_prompt, temperature=0.0)
        chosen_id = choice.get("chosen_strategy_id", strategies[0]["id"])

        # Step 2 — solve with every strategy
        results_per_strategy: dict[str, dict] = {}
        for strat in strategies:
            solve_prompt = (
                f"Problem: {problem}\n\n"
                f"Solve this problem using ONLY the following strategy:\n"
                f"  Strategy: {strat['name']} — {strat['description']}\n\n"
                f"Show your work, then give your final answer on the last line "
                f"after 'ANSWER: '."
            )
            raw = self.llm.prompt(solve_prompt, temperature=0.0)

            # Extract answer after 'ANSWER:'
            model_answer = raw.strip()
            if "ANSWER:" in raw.upper():
                model_answer = raw.upper().split("ANSWER:")[-1].strip()
                # Use original casing for check
                idx = raw.upper().rfind("ANSWER:")
                model_answer = raw[idx + len("ANSWER:"):].strip()

            correct = check_answer(model_answer, correct_answer, llm=self.llm, question=problem)
            results_per_strategy[strat["id"]] = {
                "strategy_id": strat["id"],
                "strategy_name": strat["name"],
                "model_answer": model_answer,
                "correct": correct,
            }

        chosen_correct = results_per_strategy.get(chosen_id, {}).get("correct", False)
        any_correct = any(r["correct"] for r in results_per_strategy.values())

        return {
            "problem_id": item["problem_id"],
            "chosen_strategy_id": chosen_id,
            "chosen_strategy_correct": chosen_correct,
            "any_strategy_correct": any_correct,
            "results_per_strategy": results_per_strategy,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {self.primary_metric: 0.0}

        chosen_correct = [r["chosen_strategy_correct"] for r in results]
        any_correct = [r["any_strategy_correct"] for r in results]

        strategy_hit_rate = float(np.mean(chosen_correct))
        any_solve_rate = float(np.mean(any_correct))

        # Opportunity cost: fraction of items where a correct strategy existed
        # but the chosen strategy was wrong
        opportunity_cost = float(np.mean([
            (not c) and a for c, a in zip(chosen_correct, any_correct)
        ]))

        return {
            "strategy_hit_rate": strategy_hit_rate,
            "any_solve_rate": any_solve_rate,
            "opportunity_cost": opportunity_cost,
            "n_items": len(results),
        }
