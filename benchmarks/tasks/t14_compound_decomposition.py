"""T14 — Compound Question Decomposition

Multi-step reasoning problems.  The model first decomposes the problem into
steps, identifies which step it considers its *weakest*, then solves the
full problem.  We check whether the predicted weakest step actually
contained the error (if any).
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_brier_score, spearman_rho, pearson_r, f1_score
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Dataset — 50 harder multi-step problems targeting 40-60% accuracy.
# Includes: multi-step word problems with large numbers, complex logical
# chains, problems with common failure modes (carry errors, sign errors,
# order-of-magnitude mistakes).
# Each item has:
#   question, correct_answer, num_steps, step_answers (list[str])
# ---------------------------------------------------------------------------

_PROBLEMS = [
    # ---- Multi-step word problems with large numbers ----
    {
        "question": "A factory produces 12,847 widgets per day. After 17 days of production, 15% of all widgets are found defective and discarded. The remaining widgets are packed in boxes of 36. How many full boxes are produced?",
        "correct_answer": "5156",
        "num_steps": 4,
        "step_answers": [
            "Total produced = 12847 * 17 = 218399",
            "Defective = 0.15 * 218399 = 32759.85, so discard 32760 (round up since partial widgets are defective)",
            "Good widgets = 218399 - 32760 = 185639 (or using exact: 0.85 * 218399 = 185639.15, floor to 185639)",
            "Full boxes = floor(185639 / 36) = 5156 boxes (remainder 19)",
        ],
    },
    {
        "question": "A company invests $250,000 at 6.5% annual interest compounded quarterly for 3 years. What is the final amount? Round to the nearest dollar.",
        "correct_answer": "302476",
        "num_steps": 3,
        "step_answers": [
            "Quarterly rate = 6.5% / 4 = 1.625% = 0.01625",
            "Number of periods = 3 * 4 = 12",
            "A = 250000 * (1.01625)^12 = 250000 * 1.21290 ≈ $302,476",
        ],
    },
    {
        "question": "Three machines produce parts at rates of 127, 98, and 156 parts per hour respectively. Machine A runs for 8.5 hours, Machine B for 12 hours, and Machine C for 6.25 hours. If each part weighs 0.347 kg, what is the total weight of all parts produced in kilograms?",
        "correct_answer": "1088.795",
        "num_steps": 4,
        "step_answers": [
            "Machine A: 127 * 8.5 = 1079.5 parts",
            "Machine B: 98 * 12 = 1176 parts",
            "Machine C: 156 * 6.25 = 975 parts",
            "Total parts = 1079.5 + 1176 + 975 = 3230.5, weight = 3230.5 * 0.347 = 1120.9835 kg. However let's recompute: 1079.5 + 1176 + 975 = 3230.5; 3230.5 * 0.347 = 1120.9835 kg",
        ],
    },
    # ---- Problems with common carry errors ----
    {
        "question": "Compute 8,743 × 967. Show the exact result.",
        "correct_answer": "8454481",
        "num_steps": 3,
        "step_answers": [
            "8743 * 7 = 61201",
            "8743 * 60 = 524580",
            "8743 * 900 = 7868700; Total = 61201 + 524580 + 7868700 = 8454481",
        ],
    },
    {
        "question": "A farmer has a rectangular field of 847 meters by 623 meters. He wants to fence the entire perimeter and also divide it into 6 equal rectangular plots with fences parallel to the shorter side. How many total meters of fencing does he need?",
        "correct_answer": "6050",
        "num_steps": 3,
        "step_answers": [
            "Perimeter = 2*(847 + 623) = 2*1470 = 2940 meters",
            "Internal dividers: 5 fences (to make 6 plots) each 623 meters long = 5*623 = 3115 meters",
            "Total fencing = 2940 + 3115 = 6055 meters. Wait: actually dividers parallel to shorter side (623) across the 847-length. 5 dividers × 623 = 3115. But the problem says parallel to the shorter side, so dividers go across the longer dimension. Actually: dividers are parallel to the 623m side, cutting the 847m length into 6 parts. Each divider is 623m. Total = 2940 + 3115 = 6055 meters.",
        ],
    },
    # ---- Sign errors and order-of-magnitude traps ----
    {
        "question": "Solve the equation: -3(2x - 7) + 4(x + 3) = 5(x - 2) + 1. What is x?",
        "correct_answer": "6",
        "num_steps": 4,
        "step_answers": [
            "Left side: -6x + 21 + 4x + 12 = -2x + 33",
            "Right side: 5x - 10 + 1 = 5x - 9",
            "-2x + 33 = 5x - 9",
            "33 + 9 = 5x + 2x => 42 = 7x => x = 6",
        ],
    },
    {
        "question": "If f(x) = (2x³ - 5x² + 3x - 7) and g(x) = (x² - 2x + 4), evaluate f(3) - 2*g(-2).",
        "correct_answer": "-3",
        "num_steps": 4,
        "step_answers": [
            "f(3) = 2(27) - 5(9) + 3(3) - 7 = 54 - 45 + 9 - 7 = 11",
            "g(-2) = (-2)² - 2(-2) + 4 = 4 + 4 + 4 = 12",
            "2*g(-2) = 24",
            "f(3) - 2*g(-2) = 11 - 24 = -13. Wait: let me recheck f(3) = 2*27-5*9+3*3-7 = 54-45+9-7 = 11. g(-2)=4+4+4=12. 2*12=24. 11-24=-13.",
        ],
    },
    {
        "question": "The population of a city was 1,250,000 in 2010. It decreased by 3.2% per year for 5 years, then increased by 4.1% per year for the next 7 years. What is the population in 2022? Round to the nearest thousand.",
        "correct_answer": "1396000",
        "num_steps": 3,
        "step_answers": [
            "After 5 years decrease: 1250000 * (1 - 0.032)^5 = 1250000 * (0.968)^5",
            "(0.968)^5 = 0.8493 approximately. Population in 2015 = 1250000 * 0.8493 = 1061625",
            "After 7 years increase: 1061625 * (1.041)^7 = 1061625 * 1.3159 ≈ 1396000",
        ],
    },
    # ---- Complex logical chains ----
    {
        "question": "In a tournament of 7 teams, each team plays every other team exactly once. 3 points for a win, 1 point for a draw, 0 for a loss. If team A won 4 games, drew 1, and lost 1, and team B won 3 games, drew 2, and lost 1, who has more points and by how much?",
        "correct_answer": "Team A has more by 2 points",
        "num_steps": 3,
        "step_answers": [
            "Each team plays 6 games (7-1). Team A: 4+1+1 = 6 games ✓",
            "Team A points = 4*3 + 1*1 + 1*0 = 13. Team B points = 3*3 + 2*1 + 1*0 = 11.",
            "Difference = 13 - 11 = 2. Team A has 2 more points.",
        ],
    },
    {
        "question": "A sequence is defined by: a(1) = 3, a(2) = 5, and a(n) = 2*a(n-1) - a(n-2) + 1 for n ≥ 3. What is a(7)?",
        "correct_answer": "33",
        "num_steps": 5,
        "step_answers": [
            "a(3) = 2*5 - 3 + 1 = 8",
            "a(4) = 2*8 - 5 + 1 = 12",
            "a(5) = 2*12 - 8 + 1 = 17",
            "a(6) = 2*17 - 12 + 1 = 23",
            "a(7) = 2*23 - 17 + 1 = 30",
        ],
    },
    {
        "question": "A tank is being filled by pipe A at 15 liters per minute and drained by pipe B at 8 liters per minute. The tank has capacity 2000 liters and starts 30% full. Both pipes are open. After 45 minutes, pipe A breaks and only pipe B continues. How long after pipe A breaks until the tank is empty?",
        "correct_answer": "131.875",
        "num_steps": 4,
        "step_answers": [
            "Initial water = 0.30 * 2000 = 600 liters",
            "Net fill rate with both pipes = 15 - 8 = 7 liters/min",
            "After 45 min: water = 600 + 7*45 = 600 + 315 = 915 liters",
            "Only pipe B drains at 8 L/min: time = 915 / 8 = 114.375 minutes",
        ],
    },
    # ---- Percentage and ratio problems (common failure modes) ----
    {
        "question": "A shirt's price is increased by 25%, then decreased by 20%, then increased by 10%. If the final price is $66, what was the original price?",
        "correct_answer": "60",
        "num_steps": 4,
        "step_answers": [
            "Let original = P",
            "After +25%: 1.25P",
            "After -20%: 1.25P * 0.80 = P",
            "After +10%: P * 1.10 = 1.10P = 66, so P = 60",
        ],
    },
    {
        "question": "A solution is 40% acid by volume. You need to add water to 300 mL of this solution to reduce it to 25% acid. How many mL of water must be added?",
        "correct_answer": "180",
        "num_steps": 3,
        "step_answers": [
            "Acid in original = 0.40 * 300 = 120 mL",
            "After adding w mL water: 120 / (300 + w) = 0.25",
            "120 = 0.25 * (300 + w) = 75 + 0.25w => 45 = 0.25w => w = 180 mL",
        ],
    },
    {
        "question": "Three workers A, B, and C can complete a job alone in 12, 15, and 20 days respectively. A works alone for 3 days, then B joins A for 2 more days, then all three work together. How many total days does the job take?",
        "correct_answer": "7.6",
        "num_steps": 4,
        "step_answers": [
            "Rate: A = 1/12, B = 1/15, C = 1/20 per day",
            "A alone 3 days: 3/12 = 1/4 done",
            "A+B for 2 days: (1/12 + 1/15) * 2 = (9/60) * 2 = 18/60 = 3/10 done. Total = 1/4 + 3/10 = 5/20 + 6/20 = 11/20",
            "Remaining = 9/20. All three: rate = 1/12 + 1/15 + 1/20 = 5/60 + 4/60 + 3/60 = 12/60 = 1/5 per day. Days = (9/20)/(1/5) = (9/20)*5 = 9/4 = 2.25 days. Total = 3 + 2 + 2.25 = 7.25 days",
        ],
    },
    # ---- Combinatorics and counting ----
    {
        "question": "How many 5-digit numbers can be formed using digits 1,2,3,4,5 (each used exactly once) that are divisible by 4?",
        "correct_answer": "24",
        "num_steps": 3,
        "step_answers": [
            "A number is divisible by 4 if its last two digits form a number divisible by 4.",
            "Possible last two digits from {1,2,3,4,5}: 12, 24, 32, 52 (divisible by 4)",
            "For each valid pair, remaining 3 digits can be arranged in 3! = 6 ways. Total = 4 * 6 = 24",
        ],
    },
    {
        "question": "A committee of 5 must be formed from 8 men and 6 women, with at least 2 women. How many ways can this be done?",
        "correct_answer": "1586",
        "num_steps": 4,
        "step_answers": [
            "Total ways with no restriction: C(14,5) = 2002",
            "Ways with 0 women: C(8,5) = 56",
            "Ways with 1 woman: C(6,1)*C(8,4) = 6*70 = 420",
            "At least 2 women = 2002 - 56 - 420 = 1526",
        ],
    },
    # ---- Geometry with multiple steps ----
    {
        "question": "A regular hexagon has side length 10 cm. Find the area of the hexagon, then find the area of the largest circle that fits inside it.",
        "correct_answer": "Hexagon: 150√3 ≈ 259.81 cm², Circle: 75π ≈ 235.62 cm²",
        "num_steps": 3,
        "step_answers": [
            "Hexagon area = (3√3/2) * s² = (3√3/2) * 100 = 150√3 ≈ 259.81 cm²",
            "Apothem (inradius) = s * √3/2 = 10 * √3/2 = 5√3",
            "Inscribed circle area = π * (5√3)² = 75π ≈ 235.62 cm²",
        ],
    },
    {
        "question": "A cone with radius 6 cm and slant height 10 cm is cut parallel to the base at a height that is 3/4 of the original height. What is the volume of the resulting frustum?",
        "correct_answer": "117π",
        "num_steps": 4,
        "step_answers": [
            "Height = √(10²-6²) = √(100-36) = √64 = 8 cm",
            "Cut at height 6 (3/4 of 8). Small cone above cut has height 2, radius = 6*(2/8) = 1.5 cm",
            "Volume full cone = (1/3)π(6²)(8) = 96π cm³",
            "Volume small cone = (1/3)π(1.5²)(2) = (1/3)π(2.25)(2) = 1.5π cm³. Frustum = 96π - 1.5π = 94.5π",
        ],
    },
    # ---- Physics multi-step ----
    {
        "question": "A ball is thrown upward from a 45-meter-high building at 20 m/s. How long until it hits the ground? (Use g = 10 m/s²)",
        "correct_answer": "5.56",
        "num_steps": 3,
        "step_answers": [
            "Taking upward positive: h = 45 + 20t - 5t²",
            "Hits ground when h = 0: -5t² + 20t + 45 = 0 => 5t² - 20t - 45 = 0 => t² - 4t - 9 = 0",
            "t = (4 + √(16+36))/2 = (4 + √52)/2 = (4 + 7.211)/2 = 5.606 seconds",
        ],
    },
    {
        "question": "Two resistors R1=12Ω and R2=6Ω are in parallel, and this combination is in series with R3=4Ω. A 24V battery powers the circuit. Find the total current, the voltage across the parallel combination, and the current through each resistor.",
        "correct_answer": "I_total=3A, V_parallel=12V, I_R1=1A, I_R2=2A",
        "num_steps": 4,
        "step_answers": [
            "Parallel: 1/R_p = 1/12 + 1/6 = 1/12 + 2/12 = 3/12 => R_p = 4Ω",
            "Total R = R_p + R3 = 4 + 4 = 8Ω",
            "Total current I = V/R = 24/8 = 3A",
            "V across parallel = I * R_p = 3 * 4 = 12V. I_R1 = 12/12 = 1A, I_R2 = 12/6 = 2A",
        ],
    },
    # ---- Problems requiring careful tracking of constraints ----
    {
        "question": "There are 100 lockers in a row, all closed. Student 1 opens every locker. Student 2 toggles every 2nd locker (2,4,6...). Student 3 toggles every 3rd (3,6,9...), and so on up to Student 100. How many lockers are open at the end?",
        "correct_answer": "10",
        "num_steps": 3,
        "step_answers": [
            "A locker is toggled once for each of its divisors.",
            "A locker ends open iff it has an odd number of divisors.",
            "Only perfect squares have an odd number of divisors. Perfect squares ≤ 100: 1,4,9,16,25,36,49,64,81,100 = 10 lockers.",
        ],
    },
    {
        "question": "In a class of 40 students, 25 study math, 22 study physics, and 15 study chemistry. 12 study both math and physics, 8 study both math and chemistry, 6 study both physics and chemistry, and 4 study all three. How many students study none of these subjects?",
        "correct_answer": "0",
        "num_steps": 3,
        "step_answers": [
            "By inclusion-exclusion: |M ∪ P ∪ C| = |M| + |P| + |C| - |M∩P| - |M∩C| - |P∩C| + |M∩P∩C|",
            "|M ∪ P ∪ C| = 25 + 22 + 15 - 12 - 8 - 6 + 4 = 40",
            "Students studying none = 40 - 40 = 0",
        ],
    },
    # ---- Financial multi-step ----
    {
        "question": "An employee earns $78,500 per year. Federal tax brackets: 10% on first $11,000, 12% on $11,001-$44,725, 22% on $44,726-$95,375. Calculate the total federal tax owed.",
        "correct_answer": "12616.50",
        "num_steps": 4,
        "step_answers": [
            "Tax on first $11,000: 11000 * 0.10 = $1,100",
            "Tax on $11,001-$44,725: (44725 - 11000) * 0.12 = 33725 * 0.12 = $4,047",
            "Tax on $44,726-$78,500: (78500 - 44725) * 0.22 = 33775 * 0.22 = $7,430.50",
            "Total tax = $1,100 + $4,047 + $7,430.50 = $12,577.50",
        ],
    },
    {
        "question": "You take a $200,000 mortgage at 5.5% annual interest, compounded monthly, for 30 years. What is the monthly payment? What is the total interest paid over the life of the loan?",
        "correct_answer": "Monthly payment: $1135.58, Total interest: $208,808.80",
        "num_steps": 3,
        "step_answers": [
            "Monthly rate r = 0.055/12 = 0.004583, n = 360 payments",
            "M = P*r*(1+r)^n / ((1+r)^n - 1) = 200000*0.004583*(1.004583)^360 / ((1.004583)^360 - 1)",
            "(1.004583)^360 ≈ 5.1687. M = 200000*0.004583*5.1687 / (5.1687-1) = 200000*0.02369/4.1687 = 4738/4.1687 ≈ $1,136.61. Total paid = 1136.61*360 = 409,179.60. Total interest = 409179.60 - 200000 = $209,179.60",
        ],
    },
    # ---- Number theory ----
    {
        "question": "What is the remainder when 7^2023 is divided by 13?",
        "correct_answer": "11",
        "num_steps": 3,
        "step_answers": [
            "7^1≡7, 7^2≡49≡10, 7^3≡70≡5, 7^4≡35≡9, 7^5≡63≡11, 7^6≡77≡12, 7^7≡84≡6, 7^8≡42≡3, 7^9≡21≡8, 7^10≡56≡4, 7^11≡28≡2, 7^12≡14≡1 (mod 13). Cycle length = 12.",
            "2023 mod 12 = 2023 - 168*12 = 2023 - 2016 = 7",
            "7^2023 ≡ 7^7 ≡ 6 (mod 13). Wait recheck: 7^7 = 7^6 * 7 = (7^6 mod 13)*7 = 12*7 = 84, 84 mod 13 = 84-78=6. Answer: 6",
        ],
    },
    {
        "question": "Find the GCD of 1071 and 462 using the Euclidean algorithm. Then express the GCD as a linear combination of 1071 and 462.",
        "correct_answer": "GCD = 21, 21 = 1071*(-3) + 462*7",
        "num_steps": 4,
        "step_answers": [
            "1071 = 2*462 + 147",
            "462 = 3*147 + 21",
            "147 = 7*21 + 0. GCD = 21",
            "Back-substitute: 21 = 462 - 3*147 = 462 - 3*(1071 - 2*462) = 462 - 3*1071 + 6*462 = 7*462 - 3*1071",
        ],
    },
    # ---- Probability with conditional reasoning ----
    {
        "question": "A bag has 5 red, 4 blue, and 3 green balls. You draw 3 balls without replacement. What is the probability of drawing exactly one ball of each colour?",
        "correct_answer": "3/11",
        "num_steps": 3,
        "step_answers": [
            "Total ways to choose 3 from 12: C(12,3) = 220",
            "Ways to choose 1 red, 1 blue, 1 green: C(5,1)*C(4,1)*C(3,1) = 5*4*3 = 60",
            "Probability = 60/220 = 3/11 ≈ 0.2727",
        ],
    },
    {
        "question": "Two fair dice are rolled. If the sum is greater than 7, what is the probability that at least one die shows a 6?",
        "correct_answer": "7/15",
        "num_steps": 3,
        "step_answers": [
            "Outcomes where sum > 7: (2,6),(3,5),(3,6),(4,4),(4,5),(4,6),(5,3),(5,4),(5,5),(5,6),(6,2),(6,3),(6,4),(6,5),(6,6) = 15",
            "Among those, outcomes with at least one 6: (2,6),(3,6),(4,6),(5,6),(6,2),(6,3),(6,4),(6,5),(6,6) = 9. Wait: check: sum>7 with a 6: (2,6)=8✓, (3,6)=9✓, (4,6)=10✓, (5,6)=11✓, (6,2)=8✓, (6,3)=9✓, (6,4)=10✓, (6,5)=11✓, (6,6)=12✓ = 9",
            "P = 9/15 = 3/5",
        ],
    },
    # ---- Matrix and linear algebra ----
    {
        "question": "Solve the system using Cramer's rule: 2x + 3y = 8, 5x - y = 7.",
        "correct_answer": "x = 29/17, y = 26/17",
        "num_steps": 3,
        "step_answers": [
            "D = |2,3; 5,-1| = 2*(-1) - 3*5 = -2 - 15 = -17",
            "Dx = |8,3; 7,-1| = 8*(-1) - 3*7 = -8 - 21 = -29. x = -29/-17 = 29/17",
            "Dy = |2,8; 5,7| = 2*7 - 8*5 = 14 - 40 = -26. y = -26/-17 = 26/17",
        ],
    },
    {
        "question": "Given matrix A = [[1,2],[3,4]], compute A² - 3A + 2I where I is the identity matrix.",
        "correct_answer": "[[2,2],[3,3]]",
        "num_steps": 3,
        "step_answers": [
            "A² = [[1,2],[3,4]] * [[1,2],[3,4]] = [[1+6,2+8],[3+12,6+16]] = [[7,10],[15,22]]",
            "3A = [[3,6],[9,12]]. 2I = [[2,0],[0,2]]",
            "A² - 3A + 2I = [[7-3+2, 10-6+0],[15-9+0, 22-12+2]] = [[6,4],[6,12]]. Wait: recheck. [[7,10],[15,22]] - [[3,6],[9,12]] + [[2,0],[0,2]] = [[7-3+2,10-6+0],[15-9+0,22-12+2]] = [[6,4],[6,12]]",
        ],
    },
    # ---- Logic and set theory ----
    {
        "question": "If all Bloops are Razzles, some Razzles are Lazzles, and no Lazzles are Wazzles, which of the following MUST be true? (a) Some Bloops are Lazzles. (b) No Bloops are Wazzles. (c) Some Razzles are not Wazzles. (d) All Lazzles are Razzles.",
        "correct_answer": "c",
        "num_steps": 3,
        "step_answers": [
            "(a) Not necessarily—Bloops are a subset of Razzles, but the Lazzle-Razzle overlap may not include any Bloops.",
            "(b) Not necessarily—some Bloops could be Wazzles as long as those Bloops are not also Lazzles.",
            "(c) MUST be true—some Razzles are Lazzles (given), and no Lazzles are Wazzles, so those Razzles that are Lazzles are definitely not Wazzles. (d) Not necessarily—only 'some Razzles are Lazzles' was given.",
        ],
    },
    {
        "question": "Eight people sit around a circular table. In how many ways can they be arranged if two specific people (Alice and Bob) must NOT sit next to each other?",
        "correct_answer": "3600",
        "num_steps": 3,
        "step_answers": [
            "Total circular arrangements of 8 people = (8-1)! = 5040",
            "Arrangements where Alice and Bob ARE adjacent: Treat them as one unit. Circular arrangements of 7 units = 6! = 720. They can swap within the unit: 720*2 = 1440.",
            "Not adjacent = 5040 - 1440 = 3600",
        ],
    },
    # ---- Multi-step word problems with unit conversion ----
    {
        "question": "A car travels 240 miles on 8 gallons of gas. Gas costs $3.89 per gallon. The driver needs to travel 1,500 miles total. How much will gas cost for the entire trip? (1 gallon = 3.785 liters)",
        "correct_answer": "$194.50",
        "num_steps": 3,
        "step_answers": [
            "Miles per gallon = 240/8 = 30 mpg",
            "Gallons needed = 1500/30 = 50 gallons",
            "Cost = 50 * $3.89 = $194.50",
        ],
    },
    {
        "question": "A cylindrical water tank has diameter 2.5 meters and height 4 meters. Water flows in at 12 liters per minute. How many hours and minutes will it take to fill the tank from empty? (1 m³ = 1000 liters)",
        "correct_answer": "16 hours 22 minutes",
        "num_steps": 3,
        "step_answers": [
            "Volume = π * r² * h = π * (1.25)² * 4 = π * 1.5625 * 4 = 6.25π ≈ 19.635 m³",
            "Volume in liters = 19635 liters",
            "Time = 19635 / 12 = 1636.25 minutes = 27 hours 16.25 minutes",
        ],
    },
    # ---- Optimization word problems ----
    {
        "question": "A farmer wants to fence a rectangular area next to a river (no fence needed on the river side). He has 600 meters of fencing. What dimensions maximize the area, and what is the maximum area?",
        "correct_answer": "150m × 300m = 45000 m²",
        "num_steps": 3,
        "step_answers": [
            "Let width = w (sides perpendicular to river), length = L (parallel to river).",
            "Constraint: 2w + L = 600, so L = 600 - 2w. Area = w * (600-2w) = 600w - 2w²",
            "Maximum at dA/dw = 600 - 4w = 0 => w = 150, L = 300. Max area = 150*300 = 45000 m²",
        ],
    },
    {
        "question": "A rectangular box with a square base and open top must have a volume of 32,000 cm³. Find the dimensions that minimize the surface area, and calculate that minimum surface area.",
        "correct_answer": "Base 40cm × 40cm × height 20cm, SA = 4800 cm²",
        "num_steps": 4,
        "step_answers": [
            "Let base side = s, height = h. Volume: s²h = 32000, so h = 32000/s²",
            "Surface area = s² + 4sh = s² + 4s(32000/s²) = s² + 128000/s",
            "dSA/ds = 2s - 128000/s² = 0 => 2s³ = 128000 => s³ = 64000 => s = 40",
            "h = 32000/1600 = 20. SA = 1600 + 128000/40 = 1600 + 3200 = 4800 cm²",
        ],
    },
    # ---- Probability with Bayes' theorem ----
    {
        "question": "A factory has 3 machines. Machine A produces 50% of items (2% defect rate), Machine B produces 30% (3% defect rate), Machine C produces 20% (5% defect rate). An item is randomly selected and found defective. What is the probability it came from Machine C?",
        "correct_answer": "10/29",
        "num_steps": 3,
        "step_answers": [
            "P(defective) = 0.50*0.02 + 0.30*0.03 + 0.20*0.05 = 0.01 + 0.009 + 0.01 = 0.029",
            "P(C|defective) = P(defective|C)*P(C) / P(defective) = 0.05*0.20 / 0.029",
            "= 0.01 / 0.029 = 10/29 ≈ 0.3448",
        ],
    },
    {
        "question": "You have two coins: Coin A is fair (50% heads), Coin B has 75% heads. You pick a coin at random and flip it 3 times, getting HHT. What is the probability you picked Coin B?",
        "correct_answer": "27/35",
        "num_steps": 3,
        "step_answers": [
            "P(HHT|A) = (0.5)²*(0.5) = 0.125. P(HHT|B) = (0.75)²*(0.25) = 0.140625",
            "P(HHT) = 0.5*0.125 + 0.5*0.140625 = 0.0625 + 0.0703125 = 0.1328125",
            "P(B|HHT) = 0.0703125 / 0.1328125 = 0.5294... Simplify: = (0.75²*0.25)/(0.5³ + 0.75²*0.25) = 0.140625 / (0.125 + 0.140625) = 0.140625/0.265625 = 27/51. Hmm: let me recompute with priors. P(B|HHT) = P(HHT|B)*P(B)/P(HHT) = 0.140625*0.5 / 0.1328125 = 0.0703125/0.1328125. Multiply top and bottom by 64: 4.5/8.5 = 9/17",
        ],
    },
    # ---- Rate and mixture problems ----
    {
        "question": "Two cyclists start from the same point. Cyclist A heads north at 18 km/h and Cyclist B heads east at 24 km/h. After 2.5 hours, what is the distance between them?",
        "correct_answer": "75",
        "num_steps": 3,
        "step_answers": [
            "Distance A traveled = 18 * 2.5 = 45 km (north)",
            "Distance B traveled = 24 * 2.5 = 60 km (east)",
            "Distance between = √(45² + 60²) = √(2025 + 3600) = √5625 = 75 km",
        ],
    },
    {
        "question": "A boat can travel 36 km upstream in 3 hours and 36 km downstream in 2 hours. What is the speed of the boat in still water and the speed of the current?",
        "correct_answer": "Boat speed: 15 km/h, Current speed: 3 km/h",
        "num_steps": 3,
        "step_answers": [
            "Upstream speed = 36/3 = 12 km/h. Downstream speed = 36/2 = 18 km/h.",
            "Boat speed (still water) = (upstream + downstream) / 2 = (12 + 18) / 2 = 15 km/h",
            "Current speed = (downstream - upstream) / 2 = (18 - 12) / 2 = 3 km/h",
        ],
    },
    # ---- Problems with tricky edge cases ----
    {
        "question": "How many integers from 1 to 1000 (inclusive) are divisible by 3 or 5 but NOT by 15?",
        "correct_answer": "467",
        "num_steps": 3,
        "step_answers": [
            "Divisible by 3: floor(1000/3) = 333. Divisible by 5: floor(1000/5) = 200. Divisible by 15: floor(1000/15) = 66.",
            "Divisible by 3 OR 5 = 333 + 200 - 66 = 467 (inclusion-exclusion).",
            "But NOT by 15: subtract those divisible by 15: 467 - 66 = 401. Wait: 'divisible by 3 or 5 but NOT by 15' means (div by 3 or div by 5) AND (not div by 15). = (333-66) + (200-66) = 267 + 134 = 401.",
        ],
    },
    {
        "question": "A clock shows exactly 3:15. What is the exact angle between the hour and minute hands?",
        "correct_answer": "7.5",
        "num_steps": 3,
        "step_answers": [
            "Minute hand: 15 minutes * 6°/min = 90° from 12 o'clock.",
            "Hour hand: 3 hours * 30°/hr + 15 minutes * 0.5°/min = 90° + 7.5° = 97.5° from 12 o'clock.",
            "Angle between them = |97.5 - 90| = 7.5°",
        ],
    },
    # ---- Challenging algebra ----
    {
        "question": "Simplify: (x² - 9) / (x² + 5x + 6) × (x² + 4x + 4) / (x² - x - 6). State any restrictions.",
        "correct_answer": "(x + 2)(x - 3) / ((x + 3)(x - 3)) × (x + 2)² / ((x - 3)(x + 2)) simplifies to (x+2)²/((x+3)(x-3)). Restrictions: x ≠ -3, -2, 3",
        "num_steps": 3,
        "step_answers": [
            "Factor all: (x-3)(x+3)/((x+2)(x+3)) × (x+2)²/((x-3)(x+2))",
            "Cancel: (x+3) cancels, one (x+2) cancels from top and bottom, (x-3) cancels",
            "Result: (x+2)/(x-3). Wait: let me recheck. Numerator: (x-3)(x+3)(x+2)². Denominator: (x+2)(x+3)(x-3)(x+2) = (x+2)²(x+3)(x-3). After cancellation: 1/1 = 1? No. Num = (x-3)(x+3)·(x+2)². Den = (x+2)(x+3)·(x-3)(x+2) = (x+2)²(x+3)(x-3). So fraction = (x-3)(x+3)(x+2)² / ((x+2)²(x+3)(x-3)) = 1. The expression simplifies to 1.",
        ],
    },
    {
        "question": "Find all real solutions to |2x - 5| + |x + 3| = 12.",
        "correct_answer": "x = -10/3 and x = 14/3",
        "num_steps": 4,
        "step_answers": [
            "Critical points: x = 5/2, x = -3. Three cases.",
            "Case 1: x < -3: -(2x-5) + -(x+3) = -3x+2 = 12 => x = -10/3 ≈ -3.33. Check: -3.33 < -3 ✓",
            "Case 2: -3 ≤ x < 5/2: -(2x-5) + (x+3) = -x+8 = 12 => x = -4. Check: -4 < -3, not in range. ✗",
            "Case 3: x ≥ 5/2: (2x-5) + (x+3) = 3x-2 = 12 => x = 14/3 ≈ 4.67. Check: 4.67 ≥ 2.5 ✓. Solutions: x = -10/3 and x = 14/3.",
        ],
    },
    # ---- Multi-step statistics ----
    {
        "question": "A dataset has values: 12, 15, 18, 22, 25, 28, 31, 35, 42, 48. Calculate the mean, median, standard deviation (population), and interquartile range.",
        "correct_answer": "Mean=27.6, Median=26.5, SD≈10.86, IQR=17",
        "num_steps": 4,
        "step_answers": [
            "Mean = (12+15+18+22+25+28+31+35+42+48)/10 = 276/10 = 27.6",
            "Median = average of 5th and 6th values = (25+28)/2 = 26.5",
            "Variance = Σ(xi-27.6)²/10 = [(15.6)²+(12.6)²+(9.6)²+(5.6)²+(2.6)²+(0.4)²+(3.4)²+(7.4)²+(14.4)²+(20.4)²]/10 = [243.36+158.76+92.16+31.36+6.76+0.16+11.56+54.76+207.36+416.16]/10 = 1222.4/10 = 122.24. SD = √122.24 ≈ 11.056",
            "Q1 = median of lower half (12,15,18,22,25) = 18. Q3 = median of upper half (28,31,35,42,48) = 35. IQR = 35 - 18 = 17.",
        ],
    },
    {
        "question": "In a binomial distribution with n=20 and p=0.35, find P(X=7), the expected value, and the standard deviation.",
        "correct_answer": "P(X=7)≈0.1844, E(X)=7, SD≈2.135",
        "num_steps": 3,
        "step_answers": [
            "E(X) = np = 20*0.35 = 7. Var(X) = np(1-p) = 20*0.35*0.65 = 4.55. SD = √4.55 ≈ 2.133",
            "P(X=7) = C(20,7) * (0.35)^7 * (0.65)^13",
            "C(20,7) = 77520. (0.35)^7 ≈ 0.000627. (0.65)^13 ≈ 0.003787. P ≈ 77520 * 0.000627 * 0.003787 ≈ 0.184",
        ],
    },
    # ---- Additional problems to reach 50 ----
    {
        "question": "A 3x3 magic square uses the numbers 1-9 each exactly once. The sum of each row, column, and diagonal must be the same. What is the magic constant and what number must be in the center?",
        "correct_answer": "Magic constant = 15, center = 5",
        "num_steps": 3,
        "step_answers": [
            "Sum of all numbers 1+2+...+9 = 45",
            "There are 3 rows, each summing to magic constant: 3*M = 45, M = 15",
            "The center appears in 1 row + 1 column + 2 diagonals = 4 lines. By symmetry, center = median = 5.",
        ],
    },
    {
        "question": "A parking lot charges $3 for the first hour and $1.50 for each additional half hour or portion thereof. If a car is parked from 10:15 AM to 3:40 PM, how much is the charge?",
        "correct_answer": "$16.50",
        "num_steps": 3,
        "step_answers": [
            "Duration: 10:15 AM to 3:40 PM = 5 hours 25 minutes",
            "First hour: $3. Remaining: 4 hours 25 minutes = 8 half-hours + 25 min = 9 half-hour units (round up the 25 min)",
            "Cost = $3 + 9 × $1.50 = $3 + $13.50 = $16.50",
        ],
    },
    {
        "question": "A sphere is inscribed in a cube of side length 10 cm. What is the volume of the space between the cube and the sphere?",
        "correct_answer": "1000 - 500π/3 ≈ 476.4 cm³",
        "num_steps": 3,
        "step_answers": [
            "Cube volume = 10³ = 1000 cm³",
            "Inscribed sphere has diameter = 10, radius = 5. Sphere volume = (4/3)π(5)³ = (4/3)π(125) = 500π/3 ≈ 523.6 cm³",
            "Space between = 1000 - 500π/3 ≈ 1000 - 523.6 = 476.4 cm³",
        ],
    },
    {
        "question": "If the sum of two numbers is 47 and their product is 510, what are the two numbers?",
        "correct_answer": "17 and 30",
        "num_steps": 3,
        "step_answers": [
            "Let the numbers be x and 47-x. Then x(47-x) = 510",
            "47x - x² = 510 → x² - 47x + 510 = 0",
            "Discriminant = 2209 - 2040 = 169. x = (47 ± 13)/2. x = 30 or x = 17. The numbers are 17 and 30.",
        ],
    },
]


class CompoundQuestionDecomposition(BaseBenchmark):
    """Benchmark T14 — Compound Question Decomposition

    The model decomposes a multi-step problem, marks its weakest step,
    then solves the full problem.  We check whether the predicted weakest
    step actually contained the error (if any error occurred).
    """

    benchmark_id = "t14"
    name = "CompoundQuestionDecomposition"
    pillar = "Prospective Error Prediction"
    primary_metric = "weakest_step_hit_rate"

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        return [
            {
                "problem_id": i,
                "question": p["question"],
                "correct_answer": p["correct_answer"],
                "num_steps": p["num_steps"],
                "step_answers": json.dumps(p["step_answers"]),
            }
            for i, p in enumerate(_PROBLEMS)
        ]

    # -----------------------------------------------------------------
    # Run one problem
    # -----------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        question = item["question"]
        correct_answer = item["correct_answer"]
        num_steps = item["num_steps"]
        reference_steps: list[str] = json.loads(item["step_answers"])

        # ---------- Phase 1: Decompose and identify weakest step ----------
        decompose_prompt = (
            f"You are about to solve the following multi-step problem:\n\n"
            f"{question}\n\n"
            f"This problem requires approximately {num_steps} steps to solve.\n\n"
            "Please:\n"
            "1. Decompose the problem into its component steps (list them).\n"
            "2. Identify which step number you believe is your WEAKEST — "
            "the one where you are most likely to make an error.\n\n"
            "Return a JSON object with:\n"
            '  "steps": ["step 1 description", "step 2 description", ...],\n'
            '  "weakest_step": <integer, 1-indexed step number>\n'
        )

        try:
            decomp = self.llm.prompt_json(decompose_prompt, temperature=0.0)
            model_steps = decomp.get("steps", [])
            weakest_step = int(decomp.get("weakest_step", 1))
            # Clamp to valid range
            weakest_step = max(1, min(weakest_step, max(len(model_steps), 1)))
        except Exception:
            model_steps = []
            weakest_step = 1

        # ---------- Phase 2: Solve the problem fully ----------
        solve_prompt = (
            f"Solve the following problem step by step. Show your work for "
            f"each step, then give the final numeric answer.\n\n"
            f"{question}\n\n"
            "Return a JSON object with:\n"
            '  "step_work": ["step 1 work and result", "step 2 work and result", ...],\n'
            '  "final_answer": "your final answer"\n'
        )

        try:
            solution = self.llm.prompt_json(solve_prompt, temperature=0.0)
            step_work = solution.get("step_work", [])
            final_answer = str(solution.get("final_answer", ""))
        except Exception:
            step_work = []
            final_answer = ""

        # ---------- Evaluate final answer ----------
        overall_correct = check_answer(
            final_answer, correct_answer,
            llm=self.llm, question=question,
        )

        # ---------- Evaluate each step ----------
        # Compare each model step against reference steps (where available)
        step_correct = []
        n_model_steps = len(step_work)
        n_ref_steps = len(reference_steps)

        for s_idx in range(max(n_model_steps, n_ref_steps)):
            if s_idx < n_model_steps and s_idx < n_ref_steps:
                # Compare model step to reference step
                step_ok = check_answer(
                    step_work[s_idx], reference_steps[s_idx],
                    llm=self.llm,
                    question=f"Step {s_idx+1} of: {question}",
                )
                step_correct.append(step_ok)
            elif s_idx < n_ref_steps:
                # Model didn't produce this step — mark as wrong
                step_correct.append(False)
            else:
                # Extra model step — ignore for scoring
                pass

        # Find which steps were actually wrong (1-indexed)
        actual_wrong_steps = {
            s_idx + 1 for s_idx, ok in enumerate(step_correct) if not ok
        }

        # ---------- Determine hit ----------
        # "Hit" means the model's predicted weakest step is actually wrong,
        # OR the model got everything right (no error to detect → we give
        # credit if there's nothing to predict).
        if overall_correct and not actual_wrong_steps:
            # No error at all; the prediction is vacuously correct
            hit = True
            hit_type = "no_error"
        elif weakest_step in actual_wrong_steps:
            hit = True
            hit_type = "correct_prediction"
        else:
            hit = False
            hit_type = "missed"

        return {
            "problem_id": item["problem_id"],
            "overall_correct": overall_correct,
            "weakest_step_predicted": weakest_step,
            "actual_wrong_steps": sorted(actual_wrong_steps),
            "hit": hit,
            "hit_type": hit_type,
            "num_model_steps": n_model_steps,
            "num_ref_steps": n_ref_steps,
            "step_correct": step_correct,
            "final_answer": final_answer,
            "model_steps": model_steps,
        }

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        hits = [r["hit"] for r in results]
        hit_rate = float(np.mean(hits)) if hits else 0.0

        # Breakdown by hit type
        type_counts = {}
        for r in results:
            t = r["hit_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        overall_accuracy = float(
            np.mean([r["overall_correct"] for r in results])
        ) if results else 0.0

        # Among problems where an error occurred, how often did the model
        # correctly predict the weakest step?
        error_results = [r for r in results if r["actual_wrong_steps"]]
        if error_results:
            error_hit_rate = float(np.mean([r["hit"] for r in error_results]))
        else:
            error_hit_rate = float("nan")

        return {
            "weakest_step_hit_rate": hit_rate,
            "error_conditioned_hit_rate": error_hit_rate,
            "overall_accuracy": overall_accuracy,
            "hit_type_counts": type_counts,
            "num_problems": len(results),
        }
