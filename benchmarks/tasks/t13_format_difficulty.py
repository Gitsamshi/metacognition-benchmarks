"""T13 — Format Difficulty Awareness

Tests whether a model can predict how the *format* of a question affects
its accuracy.  The same knowledge point is tested in four formats:
  1. mcq           — multiple-choice
  2. fill_blank    — fill in the blank
  3. open_qa       — open-ended question
  4. apply         — application / word-problem style

The model first predicts its per-format accuracy, then actually does each.
Score: 1 - mean Brier score across all (knowledge-point, format) pairs.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_brier_score, spearman_rho, pearson_r, f1_score
from ..llm_client import ConversationSession
import json
import numpy as np

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------
FORMATS = ["mcq", "fill_blank", "open_qa", "apply"]

# ---------------------------------------------------------------------------
# Knowledge points — 30 items, each with 4 format variants.
# Topics chosen where format matters: complex nomenclature, derivations,
# algorithm implementation.  "apply" format is genuinely hard (multi-step
# calculations, code implementation).  Some items are "format-reversed"
# where open-ended is easier than MCQ.
# ---------------------------------------------------------------------------
_KNOWLEDGE_POINTS = [
    # --- 1. IUPAC Nomenclature (complex naming where format matters) ---
    {
        "topic": "IUPAC alkane nomenclature",
        "mcq": {
            "q": "What is the IUPAC name of CH3CH(CH3)CH2CH3?\nA) 2-methylbutane\nB) 3-methylbutane\nC) isopentane\nD) neopentane",
            "a": "A", "aliases": "2-methylbutane",
        },
        "fill_blank": {
            "q": "The IUPAC name for CH3CH(CH3)CH2CH3 is ______.",
            "a": "2-methylbutane", "aliases": "",
        },
        "open_qa": {
            "q": "Draw or describe the structure of 2,3-dimethylpentane and give its molecular formula.",
            "a": "C7H16; CH3CH(CH3)CH(CH3)CH2CH3", "aliases": "C7H16",
        },
        "apply": {
            "q": "Give the IUPAC name for the compound: CH3CH2C(CH3)2CH(C2H5)CH2CH3. Identify the longest chain, all substituents, and their positions.",
            "a": "3-ethyl-4,4-dimethylhexane", "aliases": "",
        },
    },
    # --- 2. Matrix determinant (computation vs concept) ---
    {
        "topic": "Matrix determinants",
        "mcq": {
            "q": "The determinant of a 2x2 matrix [[a,b],[c,d]] is:\nA) ad + bc\nB) ad - bc\nC) ac - bd\nD) ab - cd",
            "a": "B", "aliases": "ad - bc",
        },
        "fill_blank": {
            "q": "For a 2x2 matrix [[a,b],[c,d]], the determinant is ad - ______.",
            "a": "bc", "aliases": "",
        },
        "open_qa": {
            "q": "Explain what the determinant of a matrix tells you geometrically.",
            "a": "The absolute value of the determinant gives the scale factor by which the linear transformation changes areas (2D) or volumes (3D). A negative determinant indicates orientation reversal.", "aliases": "area|volume|scaling factor",
        },
        "apply": {
            "q": "Compute the determinant of the 3x3 matrix [[2, 1, 3], [0, -1, 2], [4, 3, 1]] using cofactor expansion along the first row.",
            "a": "-24", "aliases": "",
        },
    },
    # --- 3. Thermodynamics: Gibbs free energy ---
    {
        "topic": "Gibbs free energy",
        "mcq": {
            "q": "A reaction is spontaneous at constant T and P when:\nA) ΔG > 0\nB) ΔG = 0\nC) ΔG < 0\nD) ΔH < 0 always",
            "a": "C", "aliases": "ΔG < 0|delta G < 0",
        },
        "fill_blank": {
            "q": "The Gibbs free energy equation is ΔG = ΔH - T × ______.",
            "a": "ΔS", "aliases": "delta S",
        },
        "open_qa": {
            "q": "Under what conditions of ΔH and ΔS is a reaction spontaneous at all temperatures? Explain.",
            "a": "When ΔH < 0 (exothermic) and ΔS > 0 (entropy increases), ΔG = ΔH - TΔS is always negative.", "aliases": "exothermic|entropy increase",
        },
        "apply": {
            "q": "A reaction has ΔH = -92.2 kJ/mol and ΔS = -198.7 J/(mol·K). Calculate ΔG at 298 K and determine if the reaction is spontaneous. Then find the temperature at which ΔG = 0.",
            "a": "ΔG = -92200 - 298*(-198.7) = -92200 + 59213 = -32987 J/mol ≈ -33.0 kJ/mol (spontaneous). T_eq = ΔH/ΔS = 92200/198.7 ≈ 464 K.", "aliases": "-33|464",
        },
    },
    # --- 4. Binary search algorithm ---
    {
        "topic": "Binary search algorithm",
        "mcq": {
            "q": "What is the time complexity of binary search on a sorted array?\nA) O(n)\nB) O(n log n)\nC) O(log n)\nD) O(1)",
            "a": "C", "aliases": "O(log n)",
        },
        "fill_blank": {
            "q": "Binary search requires the input array to be ______ before it can be applied.",
            "a": "sorted", "aliases": "ordered",
        },
        "open_qa": {
            "q": "Explain how binary search works and why it is more efficient than linear search.",
            "a": "Binary search repeatedly halves the search space by comparing the target to the middle element, achieving O(log n) vs O(n) for linear search.", "aliases": "halves|divide",
        },
        "apply": {
            "q": "Write a Python function that performs binary search on a sorted list and returns the index of the target, or -1 if not found. Handle edge cases including empty list and single-element list.",
            "a": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1", "aliases": "lo <= hi|mid|return -1",
        },
    },
    # --- 5. Organic chemistry: SN1 vs SN2 ---
    {
        "topic": "SN1 vs SN2 reactions",
        "mcq": {
            "q": "Which factor favours an SN2 mechanism?\nA) Tertiary substrate\nB) Polar protic solvent\nC) Strong nucleophile\nD) Stable carbocation",
            "a": "C", "aliases": "Strong nucleophile",
        },
        "fill_blank": {
            "q": "In an SN2 reaction, the nucleophile attacks from the ______ side of the leaving group, causing inversion of configuration.",
            "a": "back", "aliases": "opposite|backside",
        },
        "open_qa": {
            "q": "Compare SN1 and SN2 mechanisms in terms of kinetics, stereochemistry, and substrate preference.",
            "a": "SN1: unimolecular kinetics (rate = k[substrate]), racemisation, favoured by tertiary substrates. SN2: bimolecular kinetics (rate = k[substrate][nucleophile]), Walden inversion, favoured by primary/methyl substrates.", "aliases": "racemisation|inversion|unimolecular|bimolecular",
        },
        "apply": {
            "q": "Predict the product(s) and mechanism (SN1 or SN2) for: (R)-2-bromobutane + NaOH in DMSO. Then predict the product(s) for the same substrate with NaOH in water/ethanol. Explain stereochemical outcomes.",
            "a": "DMSO (aprotic): SN2 mechanism, gives (S)-2-butanol (inversion). Water/ethanol (protic): SN1 mechanism, gives racemic 2-butanol (mixture of R and S).", "aliases": "inversion|racemic|SN2|SN1",
        },
    },
    # --- 6. Derivative rules (chain rule application) ---
    {
        "topic": "Chain rule in calculus",
        "mcq": {
            "q": "The derivative of sin(x²) is:\nA) cos(x²)\nB) 2x·cos(x²)\nC) 2x·sin(x²)\nD) cos(2x)",
            "a": "B", "aliases": "2x·cos(x²)|2x cos(x^2)",
        },
        "fill_blank": {
            "q": "If y = f(g(x)), then dy/dx = f'(g(x)) · ______.",
            "a": "g'(x)", "aliases": "dg/dx",
        },
        "open_qa": {
            "q": "State the chain rule for differentiation and give an example of when it is needed.",
            "a": "If y = f(g(x)), then dy/dx = f'(g(x)) · g'(x). Needed for composite functions like sin(x²), e^(3x), ln(cos x).", "aliases": "composite|f'(g(x))",
        },
        "apply": {
            "q": "Find the derivative of f(x) = ln(sqrt(1 + e^(2x))). Simplify your answer completely.",
            "a": "f'(x) = e^(2x) / (1 + e^(2x)). Using chain rule: d/dx[ln(sqrt(1+e^(2x)))] = 1/(sqrt(1+e^(2x))) · 1/(2·sqrt(1+e^(2x))) · 2e^(2x) = e^(2x)/(1+e^(2x)).", "aliases": "e^(2x)/(1+e^(2x))",
        },
    },
    # --- 7. Electrochemistry: Nernst equation ---
    {
        "topic": "Nernst equation",
        "mcq": {
            "q": "The Nernst equation relates cell potential to:\nA) Temperature only\nB) Pressure only\nC) Concentration of reactants and products\nD) Electrode surface area",
            "a": "C", "aliases": "Concentration",
        },
        "fill_blank": {
            "q": "The Nernst equation: E = E° - (RT/nF) × ln(______)",
            "a": "Q", "aliases": "reaction quotient",
        },
        "open_qa": {
            "q": "Derive the Nernst equation from the relationship between Gibbs free energy and cell potential.",
            "a": "ΔG = -nFE and ΔG = ΔG° + RT·ln(Q). Combining: -nFE = -nFE° + RT·ln(Q), so E = E° - (RT/nF)·ln(Q).", "aliases": "ΔG = -nFE",
        },
        "apply": {
            "q": "A galvanic cell has E° = 1.10 V (Zn/Cu). Calculate the cell potential at 25°C when [Zn²⁺] = 0.1 M and [Cu²⁺] = 0.001 M. (n=2, F=96485 C/mol, R=8.314 J/(mol·K))",
            "a": "E = 1.10 - (0.02569/2)·ln(0.1/0.001) = 1.10 - 0.01285·ln(100) = 1.10 - 0.01285·4.605 = 1.10 - 0.0592 = 1.041 V", "aliases": "1.04|1.041",
        },
    },
    # --- 8. Genetics: Hardy-Weinberg equilibrium ---
    {
        "topic": "Hardy-Weinberg equilibrium",
        "mcq": {
            "q": "In Hardy-Weinberg equilibrium, if q = 0.3, what is p?\nA) 0.3\nB) 0.7\nC) 0.09\nD) 0.49",
            "a": "B", "aliases": "0.7",
        },
        "fill_blank": {
            "q": "The Hardy-Weinberg equation for genotype frequencies is p² + 2pq + ______ = 1.",
            "a": "q²", "aliases": "q^2",
        },
        "open_qa": {
            "q": "List and explain the five conditions required for Hardy-Weinberg equilibrium.",
            "a": "No mutation, random mating, no selection, large population size (no genetic drift), no gene flow/migration.", "aliases": "no mutation|random mating|no selection",
        },
        "apply": {
            "q": "In a population, 16% of individuals are homozygous recessive (aa) for a trait. Assuming Hardy-Weinberg equilibrium, calculate the frequencies of all three genotypes (AA, Aa, aa) and the allele frequencies (p and q). How many heterozygous carriers would you expect in a population of 10,000?",
            "a": "q² = 0.16, so q = 0.4, p = 0.6. AA = p² = 0.36, Aa = 2pq = 0.48, aa = 0.16. Heterozygous carriers in 10000: 0.48 × 10000 = 4800.", "aliases": "4800|0.48",
        },
    },
    # --- 9. Recursion: Fibonacci implementation ---
    {
        "topic": "Fibonacci sequence and recursion",
        "mcq": {
            "q": "What is the time complexity of a naive recursive Fibonacci implementation?\nA) O(n)\nB) O(n²)\nC) O(2^n)\nD) O(log n)",
            "a": "C", "aliases": "O(2^n)|exponential",
        },
        "fill_blank": {
            "q": "The technique of storing previously computed results to speed up recursive algorithms is called ______.",
            "a": "memoization", "aliases": "memoisation|dynamic programming",
        },
        "open_qa": {
            "q": "Why is naive recursive Fibonacci slow, and how does memoization improve it?",
            "a": "Naive recursion recalculates the same subproblems exponentially many times. Memoization stores each result, reducing time complexity from O(2^n) to O(n).", "aliases": "subproblems|O(n)|cache",
        },
        "apply": {
            "q": "Write a Python function that computes the nth Fibonacci number using bottom-up dynamic programming with O(1) space (no array, just two variables). The function should handle n=0 and n=1 correctly.",
            "a": "def fib(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b", "aliases": "a, b = b, a + b|O(1)",
        },
    },
    # --- 10. Electromagnetic spectrum ---
    {
        "topic": "Electromagnetic spectrum",
        "mcq": {
            "q": "Which type of EM radiation has the shortest wavelength?\nA) Radio waves\nB) Visible light\nC) X-rays\nD) Gamma rays",
            "a": "D", "aliases": "Gamma rays",
        },
        "fill_blank": {
            "q": "The relationship between wavelength (λ) and frequency (f) of electromagnetic radiation is c = λ × ______.",
            "a": "f", "aliases": "frequency|ν",
        },
        "open_qa": {
            "q": "Arrange the EM spectrum from lowest to highest energy and explain the relationship between wavelength, frequency, and energy.",
            "a": "Radio < Microwave < Infrared < Visible < UV < X-ray < Gamma. E = hf = hc/λ, so shorter wavelength means higher frequency and energy.", "aliases": "E = hf|shorter wavelength",
        },
        "apply": {
            "q": "A photon has a wavelength of 250 nm. Calculate its frequency, energy in joules, and energy in electron volts. Then determine what type of EM radiation this is. (c = 3×10⁸ m/s, h = 6.626×10⁻³⁴ J·s, 1 eV = 1.602×10⁻¹⁹ J)",
            "a": "f = c/λ = 3e8/250e-9 = 1.2e15 Hz. E = hf = 6.626e-34 × 1.2e15 = 7.95e-19 J = 4.96 eV. This is ultraviolet radiation.", "aliases": "1.2e15|4.96|ultraviolet",
        },
    },
    # --- 11. Taylor series ---
    {
        "topic": "Taylor series expansion",
        "mcq": {
            "q": "The Maclaurin series for e^x starts with:\nA) 1 + x + x²/2 + ...\nB) x - x³/6 + ...\nC) 1 - x²/2 + ...\nD) x + x³/3 + ...",
            "a": "A", "aliases": "1 + x + x²/2",
        },
        "fill_blank": {
            "q": "The general term of the Taylor series of f(x) about x=a is f⁽ⁿ⁾(a)/______ × (x-a)ⁿ.",
            "a": "n!", "aliases": "n factorial",
        },
        "open_qa": {
            "q": "What is a Taylor series and when does it converge to the original function?",
            "a": "A Taylor series represents a function as an infinite sum of terms calculated from the function's derivatives at a single point. It converges to f(x) within the radius of convergence, where the remainder term approaches zero.", "aliases": "infinite sum|derivatives|radius of convergence",
        },
        "apply": {
            "q": "Compute the Taylor series of f(x) = 1/(1-x)² about x=0 up to the x³ term. Verify by differentiating the geometric series 1/(1-x) = Σxⁿ.",
            "a": "d/dx[1/(1-x)] = 1/(1-x)² = Σ(n+1)xⁿ = 1 + 2x + 3x² + 4x³ + ...", "aliases": "1 + 2x + 3x²|Σ(n+1)xⁿ",
        },
    },
    # --- 12. Constitutional law: First Amendment (open QA easier than MCQ) ---
    {
        "topic": "First Amendment protections",
        "mcq": {
            "q": "Which is NOT protected by the First Amendment?\nA) Political speech\nB) Symbolic speech (flag burning)\nC) True threats of violence\nD) Satirical criticism of public officials",
            "a": "C", "aliases": "True threats",
        },
        "fill_blank": {
            "q": "The First Amendment protects freedom of speech, religion, press, assembly, and the right to ______ the government.",
            "a": "petition", "aliases": "",
        },
        "open_qa": {
            "q": "What freedoms does the First Amendment protect?",
            "a": "Freedom of speech, religion (free exercise and establishment clause), press, peaceable assembly, and the right to petition the government for redress of grievances.", "aliases": "speech|religion|press|assembly|petition",
        },
        "apply": {
            "q": "A public university bans all political protests on campus grounds. A student group sues. Analyse the likely outcome under First Amendment jurisprudence, considering: (a) the forum analysis (public vs limited vs nonpublic forum), (b) content neutrality, and (c) narrow tailoring. Cite relevant precedent principles.",
            "a": "University grounds are likely a designated public forum. A blanket ban on political protest is content-based restriction subject to strict scrutiny. Under Perry Education Assn v. Perry, the government must show compelling interest and narrow tailoring. A total ban is likely unconstitutional as overbroad; time/place/manner restrictions would be more defensible.", "aliases": "strict scrutiny|public forum|overbroad",
        },
    },
    # --- 13. Orbital mechanics ---
    {
        "topic": "Kepler's laws and orbital mechanics",
        "mcq": {
            "q": "Kepler's third law states that T² is proportional to:\nA) r\nB) r²\nC) r³\nD) r⁴",
            "a": "C", "aliases": "r³|r^3",
        },
        "fill_blank": {
            "q": "According to Kepler's second law, a line from a planet to the Sun sweeps out equal ______ in equal times.",
            "a": "areas", "aliases": "area",
        },
        "open_qa": {
            "q": "State Kepler's three laws of planetary motion.",
            "a": "1) Orbits are ellipses with the Sun at one focus. 2) A line from planet to Sun sweeps equal areas in equal times. 3) T² ∝ a³ (orbital period squared is proportional to semi-major axis cubed).", "aliases": "ellipses|equal areas|T² ∝ a³",
        },
        "apply": {
            "q": "Mars has an orbital period of 687 Earth days. Using Kepler's third law and Earth's orbital radius of 1 AU, calculate Mars's semi-major axis in AU. Then calculate the minimum transfer orbit time (Hohmann transfer) from Earth to Mars.",
            "a": "T²_Mars/T²_Earth = a³_Mars/a³_Earth. (687/365.25)² = a³. 3.537 = a³. a = 1.524 AU. Hohmann transfer semi-major axis = (1 + 1.524)/2 = 1.262 AU. Transfer time = T_transfer/2 = (1.262^1.5 × 365.25)/2 ≈ 259 days.", "aliases": "1.524|259",
        },
    },
    # --- 14. Signal processing: Fourier transform ---
    {
        "topic": "Fourier transform",
        "mcq": {
            "q": "The Fourier transform converts a signal from:\nA) Analog to digital\nB) Time domain to frequency domain\nC) Continuous to discrete\nD) Real to complex",
            "a": "B", "aliases": "Time domain to frequency domain",
        },
        "fill_blank": {
            "q": "The Nyquist sampling theorem states that to avoid aliasing, the sampling rate must be at least ______ times the highest frequency in the signal.",
            "a": "2", "aliases": "two|twice",
        },
        "open_qa": {
            "q": "Explain the Fourier transform intuitively and why it is useful in signal processing.",
            "a": "The Fourier transform decomposes a signal into its constituent sinusoidal frequencies, revealing the frequency content. This is useful for filtering noise, compression, spectral analysis, and solving differential equations.", "aliases": "frequency|sinusoidal|decompose",
        },
        "apply": {
            "q": "An audio signal contains frequencies at 440 Hz, 880 Hz, and 15000 Hz. You sample it at 32000 Hz. (a) Is the Nyquist criterion satisfied? (b) If you then apply a low-pass filter with cutoff at 1000 Hz, which frequency components remain? (c) What is the minimum sampling rate needed to reconstruct the original signal without aliasing?",
            "a": "(a) Yes: 32000 > 2×15000 = 30000 Hz. (b) Only 440 Hz and 880 Hz remain (below 1000 Hz cutoff). (c) Minimum sampling rate = 2 × 15000 = 30000 Hz.", "aliases": "30000|440|880",
        },
    },
    # --- 15. Protein structure levels ---
    {
        "topic": "Protein structure hierarchy",
        "mcq": {
            "q": "Alpha helices and beta sheets are examples of:\nA) Primary structure\nB) Secondary structure\nC) Tertiary structure\nD) Quaternary structure",
            "a": "B", "aliases": "Secondary structure",
        },
        "fill_blank": {
            "q": "The sequence of amino acids in a polypeptide chain is its ______ structure.",
            "a": "primary", "aliases": "",
        },
        "open_qa": {
            "q": "Describe the four levels of protein structure and the forces that stabilise each.",
            "a": "Primary: amino acid sequence (peptide bonds). Secondary: local folding into alpha helices and beta sheets (hydrogen bonds). Tertiary: 3D folding of entire polypeptide (hydrophobic interactions, disulfide bonds, ionic bonds, H-bonds). Quaternary: arrangement of multiple polypeptide subunits (same non-covalent forces as tertiary).", "aliases": "primary|secondary|tertiary|quaternary",
        },
        "apply": {
            "q": "A mutation changes a hydrophilic glutamic acid (Glu) to a hydrophobic valine (Val) in the sixth position of the beta-globin gene. Explain the molecular consequence at each structural level and identify the resulting disease.",
            "a": "This is the sickle cell mutation. Primary: Glu→Val at position 6 of beta-globin. Secondary: minimal local change. Tertiary: the hydrophobic Val creates a sticky patch on the protein surface. Quaternary: deoxygenated HbS molecules polymerize into rigid fibres via hydrophobic interactions between the Val patch and a complementary pocket on adjacent molecules, distorting red blood cells into sickle shapes. Disease: sickle cell anaemia.", "aliases": "sickle cell|HbS|polymerize",
        },
    },
    # --- 16. Bayes' theorem ---
    {
        "topic": "Bayes' theorem",
        "mcq": {
            "q": "Bayes' theorem computes:\nA) P(A and B)\nB) P(A|B) from P(B|A)\nC) P(A or B)\nD) The variance of A",
            "a": "B", "aliases": "P(A|B) from P(B|A)|posterior",
        },
        "fill_blank": {
            "q": "Bayes' theorem: P(A|B) = P(B|A) × P(A) / ______.",
            "a": "P(B)", "aliases": "",
        },
        "open_qa": {
            "q": "Explain Bayes' theorem and the concepts of prior, likelihood, and posterior.",
            "a": "P(A|B) = P(B|A)·P(A)/P(B). P(A) is the prior (initial belief), P(B|A) is the likelihood (evidence given hypothesis), P(A|B) is the posterior (updated belief after evidence). P(B) is the normalising constant.", "aliases": "prior|likelihood|posterior",
        },
        "apply": {
            "q": "A disease affects 1 in 1000 people. A test has 99% sensitivity (true positive rate) and 95% specificity (true negative rate). If a person tests positive, what is the probability they actually have the disease? Show the full calculation using Bayes' theorem.",
            "a": "P(D) = 0.001, P(+|D) = 0.99, P(+|¬D) = 0.05. P(+) = P(+|D)·P(D) + P(+|¬D)·P(¬D) = 0.99·0.001 + 0.05·0.999 = 0.00099 + 0.04995 = 0.05094. P(D|+) = 0.00099/0.05094 ≈ 0.0194 or about 1.94%.", "aliases": "1.94%|0.0194|about 2%",
        },
    },
    # --- 17. Redox reactions ---
    {
        "topic": "Oxidation-reduction reactions",
        "mcq": {
            "q": "In a redox reaction, the oxidising agent:\nA) Loses electrons\nB) Gains electrons\nC) Loses protons\nD) Gains protons",
            "a": "B", "aliases": "Gains electrons|is reduced",
        },
        "fill_blank": {
            "q": "The loss of electrons by a substance is called ______.",
            "a": "oxidation", "aliases": "",
        },
        "open_qa": {
            "q": "Define oxidation, reduction, oxidising agent, and reducing agent. How do you identify them in a reaction?",
            "a": "Oxidation is loss of electrons (increase in oxidation state). Reduction is gain of electrons (decrease in oxidation state). The oxidising agent gets reduced (gains electrons). The reducing agent gets oxidised (loses electrons). Identify by tracking oxidation state changes.", "aliases": "loss of electrons|gain of electrons",
        },
        "apply": {
            "q": "Balance the following redox reaction in acidic solution using the half-reaction method: MnO₄⁻ + Fe²⁺ → Mn²⁺ + Fe³⁺. Show both half-reactions, balance electrons, and give the final balanced equation.",
            "a": "Reduction: MnO₄⁻ + 8H⁺ + 5e⁻ → Mn²⁺ + 4H₂O. Oxidation: Fe²⁺ → Fe³⁺ + e⁻ (×5). Balanced: MnO₄⁻ + 8H⁺ + 5Fe²⁺ → Mn²⁺ + 5Fe³⁺ + 4H₂O.", "aliases": "5Fe²⁺|8H⁺|4H₂O",
        },
    },
    # --- 18. Linked list reversal (format-reversed: open QA easier) ---
    {
        "topic": "Linked list reversal",
        "mcq": {
            "q": "What is the time complexity of reversing a singly linked list iteratively?\nA) O(1)\nB) O(log n)\nC) O(n)\nD) O(n²)",
            "a": "C", "aliases": "O(n)",
        },
        "fill_blank": {
            "q": "To reverse a singly linked list iteratively, you need three pointers: prev, current, and ______.",
            "a": "next", "aliases": "next_node|temp",
        },
        "open_qa": {
            "q": "Explain the iterative approach to reversing a singly linked list.",
            "a": "Traverse the list maintaining three pointers (prev, current, next). At each step: save next node, reverse current's pointer to prev, advance prev and current. When done, prev is the new head.", "aliases": "three pointers|prev|reverse",
        },
        "apply": {
            "q": "Implement a function to reverse a singly linked list in groups of k nodes. For example, given 1->2->3->4->5 with k=3, the result should be 3->2->1->4->5. Handle the case where the remaining nodes are fewer than k (leave them as-is).",
            "a": "def reverseKGroup(head, k):\n    count, node = 0, head\n    while node and count < k:\n        node = node.next; count += 1\n    if count < k: return head\n    prev, curr = None, head\n    for _ in range(k):\n        nxt = curr.next; curr.next = prev; prev = curr; curr = nxt\n    head.next = reverseKGroup(curr, k)\n    return prev", "aliases": "reverseKGroup|count < k|return head",
        },
    },
    # --- 19. Statistical hypothesis testing ---
    {
        "topic": "Hypothesis testing (p-values)",
        "mcq": {
            "q": "A p-value of 0.03 means:\nA) There is a 3% chance the null hypothesis is true\nB) There is a 3% chance of observing data this extreme if H₀ is true\nC) 3% of the data supports H₁\nD) The effect size is 0.03",
            "a": "B", "aliases": "probability of observing data this extreme",
        },
        "fill_blank": {
            "q": "If the p-value is less than the significance level α, we ______ the null hypothesis.",
            "a": "reject", "aliases": "",
        },
        "open_qa": {
            "q": "Explain what a p-value is and common misconceptions about it.",
            "a": "A p-value is the probability of observing data at least as extreme as what was obtained, assuming the null hypothesis is true. Common misconceptions: it is NOT the probability that H₀ is true, NOT the probability of a Type I error, and a small p-value does NOT prove a large effect.", "aliases": "probability|null hypothesis|misconception",
        },
        "apply": {
            "q": "A researcher tests whether a new drug lowers blood pressure. In a study of n=25 patients, the mean reduction is 8 mmHg with standard deviation 12 mmHg. Conduct a one-sample t-test at α=0.05. Calculate the t-statistic, determine the degrees of freedom, and decide whether to reject H₀: μ=0. What is the 95% confidence interval for the mean reduction?",
            "a": "t = (8-0)/(12/√25) = 8/2.4 = 3.33. df = 24. Critical t(0.025, 24) ≈ 2.064. Since 3.33 > 2.064, reject H₀. 95% CI = 8 ± 2.064×2.4 = 8 ± 4.95 = (3.05, 12.95) mmHg.", "aliases": "3.33|reject|3.05|12.95",
        },
    },
    # --- 20. Stoichiometry ---
    {
        "topic": "Chemical stoichiometry",
        "mcq": {
            "q": "In 2H₂ + O₂ → 2H₂O, how many moles of water are produced from 3 moles of O₂?\nA) 3\nB) 6\nC) 2\nD) 1.5",
            "a": "B", "aliases": "6",
        },
        "fill_blank": {
            "q": "The reagent that is completely consumed in a reaction is called the ______ reagent.",
            "a": "limiting", "aliases": "",
        },
        "open_qa": {
            "q": "Explain what a limiting reagent is and how to identify it.",
            "a": "The limiting reagent is the reactant that is completely consumed first, determining the maximum amount of product. Identify it by computing moles of each reactant divided by its stoichiometric coefficient; the smallest ratio indicates the limiting reagent.", "aliases": "completely consumed|smallest ratio",
        },
        "apply": {
            "q": "In the reaction 4NH₃ + 5O₂ → 4NO + 6H₂O, you start with 34.0 g of NH₃ (M=17.0) and 96.0 g of O₂ (M=32.0). Determine the limiting reagent, the mass of NO (M=30.0) produced, and the mass of the excess reagent remaining.",
            "a": "Moles NH₃ = 34.0/17.0 = 2.00 mol. Moles O₂ = 96.0/32.0 = 3.00 mol. Ratio: NH₃ = 2.00/4 = 0.50, O₂ = 3.00/5 = 0.60. NH₃ is limiting. NO produced = 2.00 mol NH₃ × (4 mol NO/4 mol NH₃) = 2.00 mol NO = 60.0 g. O₂ consumed = 2.00 × (5/4) = 2.50 mol = 80.0 g. O₂ remaining = 96.0 - 80.0 = 16.0 g.", "aliases": "NH₃|60.0|16.0",
        },
    },
    # --- 21. Relativity: time dilation ---
    {
        "topic": "Special relativity: time dilation",
        "mcq": {
            "q": "As an object approaches the speed of light, a stationary observer sees its clock:\nA) Speed up\nB) Slow down\nC) Stay the same\nD) Stop completely",
            "a": "B", "aliases": "Slow down",
        },
        "fill_blank": {
            "q": "The Lorentz factor γ = 1/√(1 - ______/c²).",
            "a": "v²", "aliases": "v^2",
        },
        "open_qa": {
            "q": "Explain time dilation in special relativity. Include the formula and an intuitive explanation.",
            "a": "A moving clock runs slow relative to a stationary observer. Δt = γΔt₀ where γ = 1/√(1-v²/c²). Intuitively, since light speed is constant in all frames, a moving light clock takes longer between ticks as seen by a stationary observer.", "aliases": "γ|Lorentz|moving clock",
        },
        "apply": {
            "q": "A spacecraft travels at 0.8c relative to Earth. (a) Calculate the Lorentz factor γ. (b) If the ship's clock measures 5 years elapsed, how much time passed on Earth? (c) If the ship is 4 light-years from Earth when it starts, how long does the journey take in the ship's frame?",
            "a": "γ = 1/√(1-0.64) = 1/√0.36 = 1/0.6 = 5/3 ≈ 1.667. (b) Earth time = γ × 5 = 8.33 years. (c) In ship frame: distance is Lorentz contracted to 4/γ = 4×0.6 = 2.4 ly. Time = 2.4/0.8 = 3 years.", "aliases": "1.667|8.33|3 years",
        },
    },
    # --- 22. Graph theory: Euler paths ---
    {
        "topic": "Euler paths and circuits",
        "mcq": {
            "q": "An Euler circuit exists in an undirected graph if and only if:\nA) All vertices have odd degree\nB) All vertices have even degree\nC) Exactly two vertices have odd degree\nD) The graph is bipartite",
            "a": "B", "aliases": "All vertices have even degree",
        },
        "fill_blank": {
            "q": "A connected graph has an Euler path (but not circuit) if and only if it has exactly ______ vertices of odd degree.",
            "a": "2", "aliases": "two",
        },
        "open_qa": {
            "q": "What is the difference between an Euler path and an Euler circuit? State the conditions for each to exist.",
            "a": "An Euler path visits every edge exactly once. An Euler circuit is an Euler path that starts and ends at the same vertex. Euler circuit: all vertices have even degree, graph is connected. Euler path: exactly 0 or 2 vertices have odd degree.", "aliases": "every edge|even degree|two vertices",
        },
        "apply": {
            "q": "A graph has 6 vertices with degree sequence [2, 3, 3, 4, 4, 2]. (a) Does an Euler path exist? (b) Does an Euler circuit exist? (c) If an Euler path exists, between which vertices must it start and end? (d) How many edges does this graph have?",
            "a": "(a) Yes, exactly 2 vertices have odd degree (the degree-3 vertices). (b) No, not all degrees are even. (c) It must start and end at the two odd-degree vertices. (d) Edges = sum of degrees / 2 = (2+3+3+4+4+2)/2 = 18/2 = 9 edges.", "aliases": "9 edges|two odd-degree",
        },
    },
    # --- 23. Kirchhoff's circuit laws ---
    {
        "topic": "Kirchhoff's circuit laws",
        "mcq": {
            "q": "Kirchhoff's current law (KCL) states that:\nA) Voltage around a loop sums to zero\nB) Current into a node equals current out of the node\nC) Power is conserved in a circuit\nD) Resistance is additive in series",
            "a": "B", "aliases": "Current in equals current out",
        },
        "fill_blank": {
            "q": "Kirchhoff's voltage law (KVL) states that the sum of all voltages around any closed loop is ______.",
            "a": "zero", "aliases": "0",
        },
        "open_qa": {
            "q": "State Kirchhoff's two circuit laws and explain the physical principle behind each.",
            "a": "KCL: Total current into a node equals total current out (conservation of charge). KVL: The sum of voltage drops around any closed loop is zero (conservation of energy).", "aliases": "conservation of charge|conservation of energy",
        },
        "apply": {
            "q": "A circuit has two loops sharing a middle branch. Left loop: 12V battery in series with R1=4Ω and R3=2Ω (shared). Right loop: 6V battery in series with R2=3Ω and R3=2Ω (shared). Using Kirchhoff's laws, set up the system of equations and solve for the current through each resistor.",
            "a": "Let I1 flow clockwise in left loop, I2 clockwise in right loop. Through R3: I3 = I1 - I2 (downward). KVL left: 12 - 4·I1 - 2·(I1-I2) = 0 → 6·I1 - 2·I2 = 12. KVL right: 6 - 3·I2 - 2·(I2-I1) = 0 → -2·I1 + 5·I2 = 6. Solving: I1 = 84/26 ≈ 3.23 A, I2 = 60/26 ≈ 2.31 A, I3 = 24/26 ≈ 0.92 A.", "aliases": "3.23|2.31|0.92",
        },
    },
    # --- 24. RNA transcription and translation ---
    {
        "topic": "RNA transcription and translation",
        "mcq": {
            "q": "During transcription, the DNA template strand 3'-TACGGA-5' produces which mRNA?\nA) 5'-AUGCCU-3'\nB) 5'-ATGCCT-3'\nC) 5'-UACGGA-3'\nD) 5'-TACGGA-3'",
            "a": "A", "aliases": "AUGCCU",
        },
        "fill_blank": {
            "q": "The start codon AUG codes for the amino acid ______.",
            "a": "methionine", "aliases": "Met|M",
        },
        "open_qa": {
            "q": "Describe the central dogma of molecular biology and the roles of mRNA, tRNA, and ribosomes.",
            "a": "DNA → RNA → Protein. mRNA carries genetic information from DNA to ribosomes. tRNA brings amino acids to the ribosome, matching codons via anticodons. Ribosomes catalyse peptide bond formation to build proteins.", "aliases": "DNA|RNA|Protein|central dogma",
        },
        "apply": {
            "q": "Given the DNA template strand: 3'-TACAAAGCTCGATTT-5', determine: (a) the mRNA sequence, (b) the amino acid sequence using the standard genetic code, (c) what happens if the 7th base of the template (C) is mutated to A.",
            "a": "(a) mRNA: 5'-AUGUUUCGAGCUAAA-3'. (b) AUG=Met, UUU=Phe, CGA=Arg, GCU=Ala, AAA=Lys → Met-Phe-Arg-Ala-Lys. (c) Template C→A at position 7 changes mRNA codon from CGA to UGA (stop codon), causing premature termination: Met-Phe only.", "aliases": "Met-Phe-Arg-Ala-Lys|UGA|premature termination",
        },
    },
    # --- 25. Dimensional analysis ---
    {
        "topic": "Dimensional analysis and unit conversion",
        "mcq": {
            "q": "What is 1 atmosphere in Pascals?\nA) 100,000 Pa\nB) 101,325 Pa\nC) 760 Pa\nD) 14.7 Pa",
            "a": "B", "aliases": "101325|101,325",
        },
        "fill_blank": {
            "q": "One light-year is approximately ______ × 10¹² kilometres.",
            "a": "9.46", "aliases": "9.461|9.5",
        },
        "open_qa": {
            "q": "Explain how dimensional analysis can be used to check equations and convert units.",
            "a": "Dimensional analysis verifies that both sides of an equation have the same dimensions (units). It multiplies by conversion factors equal to 1 to change units while preserving the value. Useful for catching errors in physics equations.", "aliases": "same dimensions|conversion factors",
        },
        "apply": {
            "q": "The gravitational constant G = 6.674 × 10⁻¹¹ m³/(kg·s²). Convert G to units of cm³/(g·s²). Then verify dimensionally that the formula F = GMm/r² gives units of Newtons.",
            "a": "G = 6.674×10⁻¹¹ m³/(kg·s²). 1 m = 100 cm → m³ = 10⁶ cm³. 1 kg = 1000 g. G = 6.674×10⁻¹¹ × 10⁶/10³ cm³/(g·s²) = 6.674×10⁻⁸ cm³/(g·s²). Dimensional check: [G][M][m]/[r²] = m³/(kg·s²) × kg × kg / m² = kg·m/s² = N.", "aliases": "6.674×10⁻⁸|kg·m/s²",
        },
    },
    # --- 26. Dynamic programming: knapsack ---
    {
        "topic": "0/1 Knapsack problem",
        "mcq": {
            "q": "The 0/1 knapsack problem is solved optimally by:\nA) Greedy algorithm\nB) Dynamic programming\nC) Divide and conquer\nD) Brute force only",
            "a": "B", "aliases": "Dynamic programming",
        },
        "fill_blank": {
            "q": "The time complexity of the dynamic programming solution to 0/1 knapsack with n items and capacity W is O(______).",
            "a": "nW", "aliases": "n*W|n × W",
        },
        "open_qa": {
            "q": "Explain the 0/1 knapsack problem and why a greedy approach doesn't work for it.",
            "a": "Given items with weights and values and a capacity constraint, maximise total value without exceeding capacity. Each item is taken or not (0/1). Greedy fails because taking the highest value-to-weight item first can prevent a better overall combination.", "aliases": "maximise value|greedy fails",
        },
        "apply": {
            "q": "Items: A(weight=2, value=3), B(weight=3, value=4), C(weight=4, value=5), D(weight=5, value=7). Knapsack capacity=8. Build the DP table and find the optimal selection and total value.",
            "a": "DP table fills item by item. Optimal: B(3,4)+D(5,7) = weight 8, value 11. Or A(2,3)+C(4,5)+leftover. Checking: A+D=weight7,value10. B+D=weight8,value11. A+B+leftover: A+B=weight5,value7; can add nothing bigger. A+C=weight6,value8; +remaining nothing fits well. Optimal = B+D, value=11.", "aliases": "11|B+D|B and D",
        },
    },
    # --- 27. Acid-base chemistry (Henderson-Hasselbalch) ---
    {
        "topic": "Henderson-Hasselbalch equation",
        "mcq": {
            "q": "The Henderson-Hasselbalch equation is:\nA) pH = pKa + log([HA]/[A⁻])\nB) pH = pKa + log([A⁻]/[HA])\nC) pH = pKb + log([B]/[BH⁺])\nD) pH = -log[H⁺]",
            "a": "B", "aliases": "pH = pKa + log([A⁻]/[HA])",
        },
        "fill_blank": {
            "q": "At the half-equivalence point of a weak acid titration, pH equals ______.",
            "a": "pKa", "aliases": "",
        },
        "open_qa": {
            "q": "Derive the Henderson-Hasselbalch equation from the Ka expression and explain when it is most useful.",
            "a": "From Ka = [H⁺][A⁻]/[HA], take -log: pKa = pH - log([A⁻]/[HA]), rearrange: pH = pKa + log([A⁻]/[HA]). Most useful for buffer calculations when both acid and conjugate base are present in significant amounts.", "aliases": "Ka|buffer|log([A⁻]/[HA])",
        },
        "apply": {
            "q": "You have 500 mL of a buffer containing 0.2 M acetic acid (pKa = 4.76) and 0.15 M sodium acetate. (a) Calculate the pH. (b) If 0.01 mol of NaOH is added, what is the new pH? (c) What is the buffer capacity at the original pH?",
            "a": "(a) pH = 4.76 + log(0.15/0.2) = 4.76 + log(0.75) = 4.76 - 0.125 = 4.635. (b) NaOH converts HA→A⁻: new [HA] = (0.1-0.01)/0.5 = 0.18 M, new [A⁻] = (0.075+0.01)/0.5 = 0.17 M. pH = 4.76 + log(0.17/0.18) = 4.76 - 0.025 = 4.735. (c) Buffer capacity β = 2.303×C×Ka×[H⁺]/(Ka+[H⁺])² where C is total buffer concentration.", "aliases": "4.635|4.735",
        },
    },
    # --- 28. Quantum mechanics: uncertainty principle ---
    {
        "topic": "Heisenberg uncertainty principle",
        "mcq": {
            "q": "The Heisenberg uncertainty principle places a fundamental limit on:\nA) Measurement precision of a single variable\nB) Simultaneous precision of conjugate variables\nC) The speed of computation\nD) The number of particles in a system",
            "a": "B", "aliases": "conjugate variables|simultaneous measurement",
        },
        "fill_blank": {
            "q": "The uncertainty principle states: Δx · Δp ≥ ______/2.",
            "a": "ℏ", "aliases": "h-bar|h/(2π)|ħ",
        },
        "open_qa": {
            "q": "Explain the Heisenberg uncertainty principle. Is it a limitation of measurement instruments or a fundamental property of nature?",
            "a": "It is a fundamental property of quantum mechanics, not a measurement limitation. For conjugate variables like position and momentum, Δx·Δp ≥ ℏ/2. It arises from the wave nature of particles: a precisely defined position requires many momentum components, and vice versa.", "aliases": "fundamental|wave nature|Δx·Δp",
        },
        "apply": {
            "q": "An electron is confined to a region of width 1.0 nm (Δx = 1.0 × 10⁻⁹ m). (a) Calculate the minimum uncertainty in its momentum. (b) Calculate the minimum kinetic energy this implies. (c) Compare this energy to the thermal energy kT at room temperature (300K). (ℏ = 1.055×10⁻³⁴ J·s, mₑ = 9.109×10⁻³¹ kg, k = 1.381×10⁻²³ J/K)",
            "a": "(a) Δp ≥ ℏ/(2Δx) = 1.055×10⁻³⁴/(2×10⁻⁹) = 5.275×10⁻²⁶ kg·m/s. (b) KE = (Δp)²/(2m) = (5.275×10⁻²⁶)²/(2×9.109×10⁻³¹) = 2.783×10⁻⁵¹/1.822×10⁻³⁰ = 1.53×10⁻²¹ J. (c) kT = 1.381×10⁻²³ × 300 = 4.14×10⁻²¹ J. The confinement energy is about 37% of thermal energy.", "aliases": "5.275×10⁻²⁶|1.53×10⁻²¹",
        },
    },
    # --- 29. SQL queries ---
    {
        "topic": "SQL query writing",
        "mcq": {
            "q": "Which SQL clause is used to filter groups created by GROUP BY?\nA) WHERE\nB) HAVING\nC) FILTER\nD) GROUP FILTER",
            "a": "B", "aliases": "HAVING",
        },
        "fill_blank": {
            "q": "To combine rows from two tables based on a shared column, you use a ______ clause.",
            "a": "JOIN", "aliases": "INNER JOIN|join",
        },
        "open_qa": {
            "q": "Explain the difference between WHERE and HAVING in SQL, and when to use each.",
            "a": "WHERE filters individual rows before grouping. HAVING filters groups after GROUP BY has been applied. Use WHERE for row-level conditions, HAVING for aggregate conditions (e.g., HAVING COUNT(*) > 5).", "aliases": "before grouping|after GROUP BY|aggregate",
        },
        "apply": {
            "q": "Write a SQL query that finds the top 3 departments by average salary where the department has at least 5 employees, excluding any employee hired before 2020. Tables: employees(id, name, dept_id, salary, hire_date), departments(dept_id, dept_name). Return dept_name and avg_salary.",
            "a": "SELECT d.dept_name, AVG(e.salary) AS avg_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE e.hire_date >= '2020-01-01' GROUP BY d.dept_name HAVING COUNT(e.id) >= 5 ORDER BY avg_salary DESC LIMIT 3;", "aliases": "GROUP BY|HAVING COUNT|ORDER BY|LIMIT 3",
        },
    },
    # --- 30. Entropy in thermodynamics ---
    {
        "topic": "Thermodynamic entropy",
        "mcq": {
            "q": "The second law of thermodynamics states that in an isolated system:\nA) Entropy always decreases\nB) Entropy always increases or stays the same\nC) Energy is always conserved\nD) Temperature always increases",
            "a": "B", "aliases": "Entropy increases",
        },
        "fill_blank": {
            "q": "For a reversible process at constant temperature, ΔS = ______/T.",
            "a": "Q", "aliases": "q_rev|Q_rev|heat",
        },
        "open_qa": {
            "q": "What is entropy and why does it always increase in an isolated system?",
            "a": "Entropy is a measure of the number of microscopic configurations (disorder) consistent with a macroscopic state. It increases because systems naturally evolve toward the most probable macrostate, which has the most microstates (highest entropy). This is the second law of thermodynamics.", "aliases": "disorder|microstates|second law",
        },
        "apply": {
            "q": "Calculate the total entropy change when 500 g of water at 80°C is mixed with 500 g of water at 20°C in an insulated container. Assume c_p = 4.186 J/(g·K) and find the final temperature first. Show that ΔS_total > 0.",
            "a": "Final T = (80+20)/2 = 50°C = 323.15 K. T_hot = 353.15 K, T_cold = 293.15 K. ΔS_hot = mc_p·ln(T_f/T_hot) = 500×4.186×ln(323.15/353.15) = 2093×(-0.0888) = -185.8 J/K. ΔS_cold = 500×4.186×ln(323.15/293.15) = 2093×0.0971 = 203.2 J/K. ΔS_total = -185.8 + 203.2 = +17.4 J/K > 0.", "aliases": "17.4|50°C|ΔS_total > 0",
        },
    },
]


class FormatDifficultyAwareness(BaseBenchmark):
    """Benchmark T13 — Format Difficulty Awareness

    For each knowledge point the model predicts per-format accuracy, then
    actually answers each format variant.  Score: 1 − mean Brier score.
    """

    benchmark_id = "t13"
    name = "FormatDifficultyAwareness"
    pillar = "Prospective Error Prediction"
    primary_metric = "mean_brier_complement"

    # -----------------------------------------------------------------
    # Dataset — one item per knowledge point; each item bundles all 4 formats
    # -----------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        dataset = []
        for kp_idx, kp in enumerate(_KNOWLEDGE_POINTS):
            formats_data = {}
            for fmt in FORMATS:
                entry = kp[fmt]
                formats_data[fmt] = {
                    "question": entry["q"],
                    "answer": entry["a"],
                    "aliases": entry.get("aliases", ""),
                }
            dataset.append({
                "kp_id": kp_idx,
                "topic": kp["topic"],
                "formats_json": json.dumps(formats_data),
            })
        return dataset

    # -----------------------------------------------------------------
    # Run one knowledge point (all 4 formats)
    # -----------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        topic = item["topic"]
        formats_data: dict = json.loads(item["formats_json"])

        # ---------- Phase 1: Predict per-format accuracy ----------
        preview_lines = []
        for fmt in FORMATS:
            preview_lines.append(f"  Format '{fmt}': {formats_data[fmt]['question']}")
        preview_text = "\n".join(preview_lines)

        prediction_prompt = (
            f"You will be tested on the topic '{topic}' in four different formats.\n"
            f"Here are the four questions:\n{preview_text}\n\n"
            "For each format, predict the probability (0.0 to 1.0) that you "
            "will answer correctly.\n\n"
            "Return a JSON object with keys matching the format names and "
            "float values. Example:\n"
            '{"mcq": 0.95, "fill_blank": 0.80, "open_qa": 0.70, "apply": 0.60}'
        )

        try:
            pred_resp = self.llm.prompt_json(prediction_prompt, temperature=0.0)
            predictions = {}
            for fmt in FORMATS:
                val = pred_resp.get(fmt, 0.5)
                predictions[fmt] = float(np.clip(val, 0.0, 1.0))
        except Exception:
            predictions = {fmt: 0.5 for fmt in FORMATS}

        # ---------- Phase 2: Actually answer each format ----------
        actuals = {}
        model_answers = {}
        for fmt in FORMATS:
            entry = formats_data[fmt]
            answer_prompt = (
                f"Answer the following question concisely.\n"
                f"Question: {entry['question']}\n"
                f"Give ONLY the answer, nothing else."
            )
            resp = self.llm.prompt(answer_prompt, temperature=0.0)
            model_answers[fmt] = resp
            correct = check_answer(
                resp, entry["answer"],
                accept_aliases=entry["aliases"] if entry["aliases"] else None,
                llm=self.llm, question=entry["question"],
            )
            actuals[fmt] = 1.0 if correct else 0.0

        # ---------- Compute per-format Brier ----------
        brier_scores = {}
        for fmt in FORMATS:
            brier_scores[fmt] = (predictions[fmt] - actuals[fmt]) ** 2

        mean_brier = float(np.mean(list(brier_scores.values())))

        return {
            "kp_id": item["kp_id"],
            "topic": topic,
            "predictions": predictions,
            "actuals": actuals,
            "brier_scores": brier_scores,
            "mean_brier": mean_brier,
            "model_answers": model_answers,
        }

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        all_predicted = []
        all_actual = []
        per_format_brier = {fmt: [] for fmt in FORMATS}

        for r in results:
            for fmt in FORMATS:
                all_predicted.append(r["predictions"][fmt])
                all_actual.append(r["actuals"][fmt])
                per_format_brier[fmt].append(r["brier_scores"][fmt])

        overall_brier = compute_brier_score(all_actual, all_predicted)
        mean_brier_complement = 1.0 - overall_brier

        format_summaries = {}
        for fmt in FORMATS:
            fmt_brier = float(np.mean(per_format_brier[fmt]))
            fmt_acc = float(np.mean([r["actuals"][fmt] for r in results]))
            fmt_pred = float(np.mean([r["predictions"][fmt] for r in results]))
            format_summaries[fmt] = {
                "mean_brier": fmt_brier,
                "mean_actual_accuracy": fmt_acc,
                "mean_predicted_accuracy": fmt_pred,
            }

        return {
            "mean_brier_complement": float(np.clip(mean_brier_complement, 0.0, 1.0)),
            "overall_brier": float(overall_brier),
            "per_format": format_summaries,
            "num_knowledge_points": len(results),
            "ece": compute_ece(all_actual, all_predicted),
        }
