"""T31 — Expertise Gradient: Within each domain, questions range from
introductory (level 1) to frontier research (level 5).  Tests whether
the model's confidence curve tracks its actual accuracy curve."""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, pearson_r, spearman_rho, compute_auroc
from ..llm_client import ConversationSession
import json
import numpy as np


# 5 domains x 5 levels x 4 questions = 100 items
_QUESTIONS = {
    "organic_chemistry": {
        "domain": "Organic Chemistry",
        1: [
            {"question": "What type of bond is formed between carbon atoms in ethane?", "answer": "single covalent bond", "aliases": "sigma bond|covalent bond"},
            {"question": "What is the molecular formula of methane?", "answer": "CH4", "aliases": ""},
            {"question": "What functional group defines an alcohol?", "answer": "hydroxyl group", "aliases": "-OH|OH group"},
            {"question": "What is the difference between a single bond and a double bond?", "answer": "A single bond involves one shared pair of electrons (sigma bond), while a double bond involves two shared pairs (one sigma and one pi bond). Double bonds are shorter and stronger.", "aliases": "sigma|pi"},
        ],
        2: [
            {"question": "What is Markovnikov's rule in electrophilic addition?", "answer": "The hydrogen atom adds to the carbon with more hydrogen atoms (the less substituted carbon), and the halide adds to the more substituted carbon.", "aliases": "hydrogen adds to carbon with more hydrogens"},
            {"question": "Name the product of the reaction between ethanol and acetic acid in the presence of an acid catalyst.", "answer": "ethyl acetate", "aliases": "ethyl ethanoate"},
            {"question": "What type of isomerism do D-glucose and L-glucose exhibit?", "answer": "enantiomerism", "aliases": "enantiomers|optical isomerism|mirror image isomers"},
            {"question": "What is a condensation reaction?", "answer": "A reaction in which two molecules combine to form a larger molecule with the loss of a small molecule, usually water. Examples include ester and amide bond formation.", "aliases": "dehydration synthesis"},
        ],
        3: [
            {"question": "Explain the difference between SN1 and SN2 nucleophilic substitution mechanisms.", "answer": "SN1 is a two-step mechanism involving a carbocation intermediate and proceeds via unimolecular kinetics. SN2 is a one-step concerted mechanism with backside attack and bimolecular kinetics.", "aliases": ""},
            {"question": "What is the Diels-Alder reaction and what type of product does it form?", "answer": "A [4+2] cycloaddition between a conjugated diene and a dienophile that forms a six-membered ring (cyclohexene derivative).", "aliases": "cycloaddition"},
            {"question": "Why does benzene not undergo typical addition reactions like alkenes?", "answer": "Benzene has aromatic stability due to delocalised pi electrons, making substitution reactions thermodynamically favoured over addition.", "aliases": "aromaticity|resonance stabilization"},
            {"question": "What determines whether a nucleophilic substitution proceeds via SN1 or SN2?", "answer": "Key factors: substrate structure (tertiary favours SN1, primary/methyl favours SN2), nucleophile strength (strong nucleophile favours SN2), solvent (polar protic favours SN1, polar aprotic favours SN2), and leaving group ability (both).", "aliases": "SN1|SN2"},
        ],
        4: [
            {"question": "Explain the Woodward-Hoffmann rules for pericyclic reactions.", "answer": "They predict whether pericyclic reactions are thermally or photochemically allowed based on the symmetry of the frontier molecular orbitals (HOMO/LUMO). Conrotatory vs disrotatory modes depend on the number of electron pairs.", "aliases": "orbital symmetry"},
            {"question": "What is the Curtin-Hammett principle?", "answer": "When two interconverting conformers react to give different products, the product ratio is determined by the difference in activation energies, not by the conformer populations, provided interconversion is fast relative to product formation.", "aliases": ""},
            {"question": "Describe the mechanism of olefin metathesis and name a key catalyst.", "answer": "Olefin metathesis involves the exchange of alkylidene groups between alkenes via a metallocyclobutane intermediate. Key catalysts include Grubbs catalysts (ruthenium carbene complexes).", "aliases": "Grubbs catalyst"},
            {"question": "Explain the Wittig reaction and its stereochemical outcome.", "answer": "The Wittig reaction converts a carbonyl group to an alkene using a phosphonium ylide. Unstabilised ylides give predominantly Z (cis) alkenes via a betaine intermediate, while stabilised ylides give predominantly E (trans) alkenes. The Schlosser modification can be used to obtain E-alkenes from unstabilised ylides.", "aliases": "phosphonium ylide|Wittig"},
        ],
        5: [
            {"question": "What is the current status of the debate regarding synchronous vs. asynchronous mechanisms in Diels-Alder reactions of sterically hindered substrates?", "answer": "Computational studies suggest that while the classic Diels-Alder is concerted and synchronous, sterically demanding substrates can proceed through highly asynchronous transition states. The boundary between concerted-asynchronous and stepwise diradical pathways remains an active area of research.", "aliases": ""},
            {"question": "Describe the challenges of enantioselective C-H functionalisation in complex natural product synthesis.", "answer": "Key challenges include controlling site-selectivity among multiple C-H bonds, achieving high enantioselectivity with catalytic directing groups, and maintaining functional group compatibility. Recent advances use palladium, rhodium, and iridium catalysts with chiral ligands.", "aliases": ""},
            {"question": "What is the significance of non-covalent interactions in organocatalysis and how are they currently modelled?", "answer": "Hydrogen bonding, pi-stacking, and ion-pairing interactions are crucial for selectivity in organocatalysis. Modern approaches use DFT calculations with dispersion corrections and non-covalent interaction (NCI) analysis to model these, though quantitative prediction remains challenging.", "aliases": ""},
            {"question": "Discuss the challenges and recent advances in late-stage functionalisation of complex molecules.", "answer": "Late-stage functionalisation aims to modify complex molecules (e.g., drug candidates) at specific C-H bonds without affecting other functional groups. Challenges include selectivity among multiple similar C-H bonds, functional group tolerance, and scalability. Recent advances use directed C-H activation with Pd/Rh/Ir catalysts, radical relay strategies, and electrochemical methods. Machine learning is increasingly used to predict site selectivity.", "aliases": ""},
        ],
    },
    "machine_learning": {
        "domain": "Machine Learning",
        1: [
            {"question": "What is the difference between supervised and unsupervised learning?", "answer": "Supervised learning uses labelled data to learn a mapping from inputs to outputs, while unsupervised learning finds patterns in data without labels.", "aliases": ""},
            {"question": "What is overfitting in machine learning?", "answer": "When a model learns the training data too well, including noise, and performs poorly on unseen data.", "aliases": ""},
            {"question": "What is the purpose of a test set?", "answer": "To evaluate model performance on data it has not seen during training, providing an unbiased estimate of generalisation.", "aliases": ""},
            {"question": "What is a neural network?", "answer": "A computational model inspired by biological neurons, consisting of layers of interconnected nodes that learn to map inputs to outputs by adjusting connection weights during training.", "aliases": ""},
        ],
        2: [
            {"question": "Explain the bias-variance tradeoff.", "answer": "Bias is error from oversimplified assumptions; variance is error from sensitivity to training data fluctuations. Reducing one tends to increase the other. The goal is to find the optimal balance that minimises total error.", "aliases": ""},
            {"question": "What is gradient descent and why is a learning rate important?", "answer": "Gradient descent is an optimisation algorithm that iteratively adjusts parameters in the direction of steepest loss decrease. The learning rate controls step size—too large causes divergence, too small causes slow convergence.", "aliases": ""},
            {"question": "What is regularisation and give an example?", "answer": "Regularisation adds a penalty term to the loss function to prevent overfitting. Examples include L1 (Lasso) which promotes sparsity and L2 (Ridge) which penalises large weights.", "aliases": ""},
            {"question": "What is cross-validation and why is it used?", "answer": "Cross-validation splits data into k folds, training on k-1 and validating on the remaining fold, rotating through all folds. It provides a more robust estimate of model performance than a single train/test split, especially with limited data.", "aliases": "k-fold"},
        ],
        3: [
            {"question": "Explain the attention mechanism in transformers.", "answer": "Attention computes a weighted sum of value vectors, where weights are determined by the compatibility (dot product) between query and key vectors, scaled and softmaxed. This allows the model to focus on relevant parts of the input regardless of position.", "aliases": ""},
            {"question": "What is batch normalisation and why does it help training?", "answer": "Batch normalisation normalises layer inputs across the mini-batch to have zero mean and unit variance, then applies learnable scale and shift. It helps by reducing internal covariate shift, allowing higher learning rates and acting as a regulariser.", "aliases": ""},
            {"question": "Explain the reparameterisation trick in variational autoencoders.", "answer": "Instead of sampling directly from the latent distribution (which is not differentiable), we sample from a standard normal and transform using the learned mean and variance. This allows gradients to flow through the sampling step.", "aliases": ""},
            {"question": "What is residual learning and why do skip connections help deep networks?", "answer": "Residual learning (ResNets) adds skip connections that allow the input to bypass one or more layers: H(x) = F(x) + x. This helps because learning the residual F(x) is easier than learning the full mapping H(x), mitigates vanishing gradients, and enables training of very deep networks (100+ layers).", "aliases": "ResNet|skip connection"},
        ],
        4: [
            {"question": "Explain the theoretical foundations of neural tangent kernels and their implications.", "answer": "In the infinite-width limit, neural networks behave like kernel methods with a fixed kernel (the NTK). Training dynamics become linear, and the model converges to the global minimum. This provides theoretical tractability but may not explain finite-width network behaviour.", "aliases": "NTK"},
            {"question": "What is the lottery ticket hypothesis and what are its practical implications?", "answer": "It posits that dense randomly-initialised networks contain sparse subnetworks (winning tickets) that, when trained in isolation from their original initialisation, can match full network performance. This has implications for network pruning and efficient training.", "aliases": ""},
            {"question": "Describe the grokking phenomenon and current explanations for it.", "answer": "Grokking is delayed generalisation where a model memorises training data first, then suddenly generalises long after achieving zero training loss. Explanations involve competition between memorisation and generalisation circuits, with weight decay eventually favouring simpler solutions.", "aliases": ""},
            {"question": "Explain the distinction between aleatoric and epistemic uncertainty in Bayesian deep learning.", "answer": "Aleatoric uncertainty is inherent noise in the data (irreducible), while epistemic uncertainty reflects model ignorance (reducible with more data). Bayesian methods (MC dropout, deep ensembles, variational inference) can separately estimate both. Epistemic uncertainty decreases with more training data; aleatoric does not.", "aliases": "aleatoric|epistemic"},
        ],
        5: [
            {"question": "What is the current understanding of why in-context learning emerges in large language models and how it relates to meta-learning?", "answer": "In-context learning may arise because transformers implicitly implement learning algorithms in their forward pass. Some work shows transformers can implement gradient descent in their attention layers. The connection to meta-learning is that pre-training on diverse tasks creates an implicit meta-learner, though the precise mechanism remains debated.", "aliases": ""},
            {"question": "Discuss the theoretical limitations of RLHF and proposed alternatives for aligning language models.", "answer": "RLHF suffers from reward model misspecification, reward hacking, and distributional shift. The Bradley-Terry preference model makes strong assumptions. Alternatives include DPO (direct preference optimisation), constitutional AI, and debate-based approaches. Open questions remain about scalable oversight.", "aliases": ""},
            {"question": "What is the current state of understanding regarding scaling laws and emergent abilities in large language models?", "answer": "Scaling laws predict smooth improvements with scale, but some abilities appear to emerge suddenly at specific scales. Recent work debates whether emergence is a genuine phase transition or an artefact of evaluation metrics. The predictability of capabilities from scaling curves remains an active research question.", "aliases": ""},
            {"question": "Discuss the theoretical relationship between diffusion models, score matching, and stochastic differential equations.", "answer": "Diffusion models add noise to data through a forward SDE and learn to reverse it. Score matching trains a neural network to estimate the score function (gradient of log-density). Song et al. unified DDPM and score-based models as discretisations of continuous-time SDEs (VP-SDE and VE-SDE). The reverse SDE is determined by the score, enabling exact likelihood computation. Probability flow ODEs provide a deterministic counterpart. Open theoretical questions include optimal noise schedules and the gap between continuous-time analysis and discrete implementations.", "aliases": ""},
        ],
    },
    "number_theory": {
        "domain": "Number Theory",
        1: [
            {"question": "What is a prime number?", "answer": "A natural number greater than 1 that has no positive divisors other than 1 and itself.", "aliases": ""},
            {"question": "What is the fundamental theorem of arithmetic?", "answer": "Every integer greater than 1 can be uniquely represented as a product of prime numbers, up to the order of the factors.", "aliases": "unique prime factorisation"},
            {"question": "Is 1 a prime number?", "answer": "No, 1 is not considered a prime number. By convention and definition, primes must be greater than 1.", "aliases": "no"},
            {"question": "What is the greatest common divisor (GCD) of two numbers?", "answer": "The largest positive integer that divides both numbers without a remainder.", "aliases": ""},
        ],
        2: [
            {"question": "State and explain the division algorithm.", "answer": "For any integers a and b with b > 0, there exist unique integers q (quotient) and r (remainder) such that a = bq + r and 0 ≤ r < b.", "aliases": "a = bq + r"},
            {"question": "What is modular arithmetic and what does 'a ≡ b (mod n)' mean?", "answer": "Modular arithmetic is a system where numbers 'wrap around' after reaching a certain value (the modulus). a ≡ b (mod n) means n divides (a - b), i.e., a and b have the same remainder when divided by n.", "aliases": "congruence"},
            {"question": "State Fermat's Little Theorem.", "answer": "If p is prime and a is not divisible by p, then a^(p-1) ≡ 1 (mod p).", "aliases": "a^(p-1) ≡ 1"},
            {"question": "What is the Euclidean algorithm?", "answer": "An efficient method for computing the GCD of two numbers by repeatedly dividing and taking remainders: GCD(a,b) = GCD(b, a mod b), terminating when the remainder is 0.", "aliases": "GCD(a,b) = GCD(b, a mod b)"},
        ],
        3: [
            {"question": "State and prove Wilson's theorem.", "answer": "Wilson's theorem: (p-1)! ≡ -1 (mod p) if and only if p is prime. Proof sketch: for prime p, each element 1..p-1 has a unique inverse mod p. Elements pair with their inverses except 1 and p-1 (which are self-inverse), so the product of all is 1·(p-1)·(paired products) = (p-1) ≡ -1 (mod p).", "aliases": "(p-1)! ≡ -1 (mod p)"},
            {"question": "Explain the Chinese Remainder Theorem and give an example.", "answer": "If n1, n2, ..., nk are pairwise coprime, then for any integers a1, ..., ak, the system x ≡ ai (mod ni) has a unique solution modulo N = n1·n2·...·nk. Example: x ≡ 2 (mod 3), x ≡ 3 (mod 5) has solution x ≡ 8 (mod 15).", "aliases": "CRT"},
            {"question": "What is a Mersenne prime and why are they significant?", "answer": "A Mersenne prime is a prime of the form 2^p - 1, where p itself is prime. They are significant because they are connected to perfect numbers (via the Euclid-Euler theorem) and are the focus of the Great Internet Mersenne Prime Search (GIMPS).", "aliases": "2^p - 1"},
            {"question": "Explain quadratic residues and the Legendre symbol.", "answer": "An integer a is a quadratic residue mod p if there exists x such that x² ≡ a (mod p). The Legendre symbol (a/p) equals 1 if a is a QR mod p, -1 if not, and 0 if p divides a. Euler's criterion: (a/p) ≡ a^((p-1)/2) (mod p).", "aliases": "Legendre symbol"},
        ],
        4: [
            {"question": "State the law of quadratic reciprocity and explain its significance.", "answer": "For distinct odd primes p and q: (p/q)(q/p) = (-1)^((p-1)/2 · (q-1)/2). This means (p/q) = (q/p) unless both p and q are ≡ 3 (mod 4). It allows determining whether p is a QR mod q by instead checking whether q is a QR mod p, which may be easier.", "aliases": "quadratic reciprocity"},
            {"question": "Explain the p-adic numbers and their significance in number theory.", "answer": "The p-adic numbers Qp complete the rationals with respect to the p-adic absolute value |x|_p = p^(-v_p(x)). They provide an alternative to real numbers where 'closeness' is determined by divisibility by powers of p. They are fundamental in algebraic number theory and the local-global principle (Hasse-Minkowski theorem).", "aliases": "p-adic"},
            {"question": "What is the Riemann zeta function and how does it relate to prime numbers?", "answer": "ζ(s) = Σ 1/n^s for Re(s) > 1. Euler showed ζ(s) = Π (1-p^(-s))^(-1) over all primes p, connecting the zeta function to prime distribution. The Prime Number Theorem relates to the non-vanishing of ζ on the line Re(s) = 1.", "aliases": "Euler product"},
            {"question": "Describe Hensel's lemma and its applications.", "answer": "Hensel's lemma provides conditions under which a solution of a polynomial equation modulo a prime p can be 'lifted' to a solution modulo higher powers of p. If f(a) ≡ 0 (mod p) and f'(a) ≢ 0 (mod p), then there exists a unique lift to Z_p. It is essential for studying p-adic equations.", "aliases": "Hensel lifting"},
        ],
        5: [
            {"question": "What is the current status of the ABC conjecture and Mochizuki's claimed proof?", "answer": "The ABC conjecture relates the prime factors of three integers a+b=c. Mochizuki claimed a proof using Inter-universal Teichmüller Theory (IUT) in 2012, published in PRIMS in 2021. However, many leading number theorists (including Scholze and Stix) identified what they consider a fundamental gap in Corollary 3.12. The mathematical community remains largely unconvinced, making this one of the most controversial episodes in modern mathematics.", "aliases": ""},
            {"question": "Describe the Langlands programme and its significance for modern number theory.", "answer": "The Langlands programme is a vast web of conjectures connecting number theory, algebraic geometry, and representation theory. It posits deep relationships between automorphic forms and Galois representations. Key results include the proof of the Taniyama-Shimura-Weil conjecture (used in Fermat's Last Theorem) and the fundamental lemma (proved by Ngô). Many core conjectures remain open.", "aliases": "Langlands"},
            {"question": "What are the major approaches to the Riemann Hypothesis and why has it resisted proof?", "answer": "Approaches include analytic methods (zero-free regions, explicit formulae), algebraic methods (analogy with the Weil conjectures proved for function fields), random matrix theory connections, and spectral theory approaches. It resists proof partly because the known zero-free region grows too slowly, and the function field analogue proof by Deligne uses algebraic geometry techniques that don't directly transfer to the number field case.", "aliases": "RH"},
            {"question": "Discuss the current understanding of gaps between consecutive primes and recent breakthroughs.", "answer": "Zhang (2013) proved bounded gaps: infinitely many prime pairs with gap < 70 million. The Polymath project reduced this to 246. The twin prime conjecture (gap = 2) remains open. Maynard-Tao independently proved bounded gaps using a different sieve method. On the other end, Rankin's theorem shows gaps can be as large as c·log(n)·loglog(n)·loglogloglog(n)/logloglog(n)² for some constant c. The distribution of gaps relates to the Hardy-Littlewood conjectures.", "aliases": ""},
        ],
    },
    "constitutional_law": {
        "domain": "Constitutional Law",
        1: [
            {"question": "What are the three branches of the US federal government?", "answer": "Legislative (Congress), Executive (President), and Judicial (Supreme Court and federal courts).", "aliases": "legislative|executive|judicial"},
            {"question": "What is the Bill of Rights?", "answer": "The first ten amendments to the US Constitution, ratified in 1791, guaranteeing fundamental rights such as freedom of speech, religion, and the right to bear arms.", "aliases": "first ten amendments"},
            {"question": "What does the concept of 'separation of powers' mean?", "answer": "The division of government responsibilities into distinct branches to prevent any one branch from exercising the core functions of another, providing a system of checks and balances.", "aliases": "checks and balances"},
            {"question": "What is the supremacy clause?", "answer": "Article VI, Clause 2 of the US Constitution, which establishes that the Constitution, federal laws, and treaties are the supreme law of the land, taking precedence over state laws.", "aliases": "Article VI"},
        ],
        2: [
            {"question": "What is judicial review and which case established it?", "answer": "Judicial review is the power of courts to declare legislative and executive acts unconstitutional. It was established in Marbury v. Madison (1803) by Chief Justice John Marshall.", "aliases": "Marbury v. Madison"},
            {"question": "What is the difference between strict scrutiny, intermediate scrutiny, and rational basis review?", "answer": "Strict scrutiny (for race, fundamental rights): government must show a compelling interest and narrow tailoring. Intermediate scrutiny (for gender): must serve an important interest and be substantially related. Rational basis: must be rationally related to a legitimate government interest.", "aliases": "levels of scrutiny"},
            {"question": "Explain the incorporation doctrine.", "answer": "The incorporation doctrine applies the Bill of Rights to state and local governments through the 14th Amendment's Due Process Clause. Most rights have been selectively incorporated through Supreme Court decisions.", "aliases": "14th Amendment|selective incorporation"},
            {"question": "What is the commerce clause and why is it significant?", "answer": "Article I, Section 8, Clause 3 grants Congress power to regulate commerce among the states. It has been broadly interpreted to give Congress extensive regulatory power, as established in cases like Wickard v. Filburn and Gonzales v. Raich.", "aliases": "Article I Section 8"},
        ],
        3: [
            {"question": "Explain the state action doctrine and its exceptions.", "answer": "The state action doctrine holds that the Constitution only restricts government action, not private conduct. Exceptions include the public function doctrine (private entities performing traditionally governmental functions), the entanglement/nexus test (significant government involvement), and the compulsion test.", "aliases": "state action"},
            {"question": "What is substantive due process and how has it evolved?", "answer": "Substantive due process holds that certain fundamental rights are protected from government interference regardless of the process provided. It evolved from Lochner-era economic liberty through its modern form protecting privacy rights (Griswold), reproductive rights (Roe/Casey/Dobbs), and intimate relationships (Lawrence, Obergefell).", "aliases": "fundamental rights"},
            {"question": "Explain the dormant commerce clause doctrine.", "answer": "Even when Congress has not acted, the Commerce Clause implicitly limits state power to regulate interstate commerce. States cannot discriminate against interstate commerce or impose undue burdens on it. The Pike balancing test weighs local benefits against burden on interstate commerce.", "aliases": "Pike balancing"},
            {"question": "What is the political question doctrine?", "answer": "Under Baker v. Carr (1962), courts may decline to adjudicate certain constitutional questions that are committed to other branches or lack judicially manageable standards. Classic examples include questions about the guarantee clause, foreign affairs powers, and impeachment procedures.", "aliases": "Baker v. Carr"},
        ],
        4: [
            {"question": "Analyse the development of the non-delegation doctrine and its potential revival.", "answer": "The non-delegation doctrine holds that Congress cannot delegate its legislative power. After Panama Refining and Schechter Poultry (1935), the Court has not struck down a statute on non-delegation grounds, accepting broad delegations with an 'intelligible principle.' Recent opinions by Justices Gorsuch and Kavanaugh suggest potential revival, particularly in the context of major questions doctrine and administrative state skepticism.", "aliases": "non-delegation|intelligible principle"},
            {"question": "Explain the major questions doctrine as articulated in West Virginia v. EPA.", "answer": "In West Virginia v. EPA (2022), the Court held that agencies cannot claim authority over issues of vast economic and political significance without clear congressional authorisation. This extends the clear statement canon to administrative delegation, requiring Congress to 'speak clearly' when authorising transformative agency action. Critics argue it lacks a principled framework for identifying 'major' questions.", "aliases": "West Virginia v. EPA"},
            {"question": "Trace the evolution of Second Amendment jurisprudence from Miller to Bruen.", "answer": "US v. Miller (1939) upheld regulation, emphasising militia use. District of Columbia v. Heller (2008) established an individual right to bear arms. McDonald v. Chicago (2010) incorporated it against states. Bruen (2022) replaced means-end scrutiny with a historical tradition test, requiring gun regulations to have analogues in American history. This represents a significant methodological shift.", "aliases": "Heller|Bruen"},
            {"question": "What is the anti-commandeering doctrine and its constitutional basis?", "answer": "The anti-commandeering doctrine prohibits the federal government from compelling state legislatures to enact regulations (New York v. US, 1992) or directing state officers to enforce federal law (Printz v. US, 1997). It is grounded in the 10th Amendment and structural federalism. Murphy v. NCAA (2018) extended it to prohibitions on state action.", "aliases": "New York v. US|Printz"},
        ],
        5: [
            {"question": "Assess the constitutional implications of the unitary executive theory in its strongest form.", "answer": "The strong unitary executive theory claims the President has complete control over all executive functions, including removal of any executive officer and directive authority over all executive agencies. This challenges independent agencies, special counsels, and statutory restrictions on removal. The debate involves Article II's Vesting Clause, historical practice, and tensions between democratic accountability and expert independence. Seila Law (2020) and Collins v. Yellen (2021) moved toward stronger presidential removal power.", "aliases": ""},
            {"question": "Discuss the current constitutional debates around Section 3 of the 14th Amendment and its application.", "answer": "Section 3 disqualifies from office those who 'engaged in insurrection' after taking an oath. Key unresolved questions after Trump v. Anderson (2024) include: whether it is self-executing, who determines 'insurrection,' whether Congress must act via legislation (as the Court held), and the historical scope (originally targeting ex-Confederates). The Court's emphasis on Congressional enforcement may have effectively narrowed the provision's application.", "aliases": "14th Amendment Section 3"},
            {"question": "Analyse the tension between originalism and living constitutionalism in modern constitutional interpretation.", "answer": "Originalism (public meaning or original intent) claims constitutional meaning is fixed at ratification, providing constraint and legitimacy. Living constitutionalism holds that meaning evolves with societal change. Modern tensions include: originalism's difficulty with abstract clauses (due process, equal protection), the problem of expected applications vs. semantic meaning, the role of precedent, and whether either framework actually constrains judicial discretion. Pragmatic originalists like Scalia accepted precedent; new originalists engage with construction zones.", "aliases": ""},
            {"question": "What are the constitutional implications of algorithmic governance and AI decision-making for due process?", "answer": "AI governance raises novel due process concerns: procedural due process requires notice and an opportunity to be heard, but opaque algorithms resist meaningful explanation. Substantive due process may be implicated when algorithms make high-stakes decisions (benefits, sentencing). Equal protection is challenged by algorithmic bias. Current doctrine lacks frameworks for evaluating algorithmic transparency, though Mathews v. Eldridge balancing could adapt. The question of whether algorithmic decisions constitute 'state action' when performed by private contractors adds complexity.", "aliases": ""},
        ],
    },
    "astrophysics": {
        "domain": "Astrophysics",
        1: [
            {"question": "What is a star and what keeps it from collapsing?", "answer": "A star is a massive ball of hot gas undergoing nuclear fusion. It is kept from collapsing by the outward radiation pressure and thermal pressure from fusion reactions balancing the inward pull of gravity.", "aliases": "hydrostatic equilibrium"},
            {"question": "What is a light-year?", "answer": "A light-year is the distance that light travels in one year, approximately 9.46 trillion kilometres (9.46 × 10^12 km).", "aliases": "9.46 trillion km"},
            {"question": "What is the difference between a planet and a star?", "answer": "A star generates energy through nuclear fusion in its core, while a planet does not. Stars are typically much more massive than planets. The boundary is roughly at about 80 Jupiter masses, above which hydrogen fusion can be sustained.", "aliases": "nuclear fusion"},
            {"question": "What is the Sun's spectral type?", "answer": "The Sun is a G2V main-sequence star, also called a yellow dwarf.", "aliases": "G2V|yellow dwarf"},
        ],
        2: [
            {"question": "Describe the Hertzsprung-Russell diagram and what it tells us.", "answer": "The HR diagram plots stellar luminosity against surface temperature (or spectral type). Stars cluster into groups: the main sequence (where most stars spend most of their lives), red giants, supergiants, and white dwarfs. Position on the diagram indicates a star's evolutionary stage.", "aliases": "HR diagram"},
            {"question": "What is the Chandrasekhar limit and why is it important?", "answer": "The Chandrasekhar limit (~1.4 solar masses) is the maximum mass of a stable white dwarf supported by electron degeneracy pressure. Above this limit, a white dwarf will collapse further, potentially triggering a Type Ia supernova or forming a neutron star.", "aliases": "1.4 solar masses"},
            {"question": "Explain the difference between Type Ia and Type II supernovae.", "answer": "Type Ia supernovae occur in binary systems when a white dwarf accretes matter exceeding the Chandrasekhar limit, causing thermonuclear explosion. Type II supernovae result from core collapse of massive stars (>8 solar masses) when fusion can no longer support the core. Type Ia have no hydrogen lines; Type II do.", "aliases": ""},
            {"question": "What evidence supports the Big Bang theory?", "answer": "Key evidence: (1) Hubble's observation of the expansion of the universe, (2) the cosmic microwave background radiation (CMB) at ~2.7 K, (3) the observed abundance of light elements (hydrogen, helium, deuterium) matching Big Bang nucleosynthesis predictions, and (4) the evolution of galaxy populations over cosmic time.", "aliases": "CMB|expansion|nucleosynthesis"},
        ],
        3: [
            {"question": "Explain the CNO cycle and how it differs from the proton-proton chain.", "answer": "Both are hydrogen fusion pathways. The pp chain dominates in stars like the Sun (T < 17 million K), directly fusing protons. The CNO cycle uses carbon, nitrogen, and oxygen as catalysts and dominates in more massive, hotter stars. The CNO cycle has a steeper temperature dependence (~T^16 vs ~T^4 for pp), making it dominant above ~1.3 solar masses.", "aliases": "CNO|proton-proton"},
            {"question": "What is gravitational lensing and how is it used in astrophysics?", "answer": "Gravitational lensing occurs when a massive object bends light from a more distant source, as predicted by general relativity. Strong lensing produces multiple images or arcs. Weak lensing causes subtle shape distortions used to map dark matter distribution. Microlensing temporarily brightens background stars, used to detect exoplanets and MACHOs.", "aliases": ""},
            {"question": "Describe the process of neutron star formation and explain what determines whether a neutron star or black hole forms.", "answer": "When a massive star's iron core exceeds the Chandrasekhar limit, electron degeneracy pressure fails and the core collapses. Protons and electrons combine into neutrons (neutronisation). If the remnant mass is below the Tolman-Oppenheimer-Volkoff limit (~2-3 solar masses), neutron degeneracy pressure halts collapse, forming a neutron star. Above this limit, collapse continues to form a black hole.", "aliases": "TOV limit"},
            {"question": "What is the cosmic microwave background and what information does it contain?", "answer": "The CMB is thermal radiation from ~380,000 years after the Big Bang, when the universe cooled enough for electrons and protons to form neutral hydrogen (recombination). Its nearly uniform ~2.7 K blackbody spectrum with tiny anisotropies (1 part in 100,000) encodes information about the early universe's density fluctuations, geometry (flat), composition (matter, dark matter, dark energy), and expansion rate.", "aliases": "CMB|recombination"},
        ],
        4: [
            {"question": "Explain the Kerr metric and its astrophysical implications.", "answer": "The Kerr metric describes the spacetime geometry around a rotating (uncharged) black hole. Unlike the Schwarzschild metric (non-rotating), it features an ergosphere where frame-dragging forces all objects to co-rotate. The Penrose process can extract rotational energy from the ergosphere. The inner and outer horizons, ring singularity, and closed timelike curves are distinctive features.", "aliases": "frame-dragging|ergosphere"},
            {"question": "What is the role of baryonic acoustic oscillations (BAO) in cosmology?", "answer": "BAO are regular periodic fluctuations in the density of visible baryonic matter caused by acoustic density waves in the primordial plasma. They create a characteristic scale (~490 million light-years today) imprinted in galaxy clustering. This 'standard ruler' allows precise measurement of the expansion history, constraining dark energy properties and the Hubble constant.", "aliases": "BAO|standard ruler"},
            {"question": "Describe r-process nucleosynthesis and its astrophysical sites.", "answer": "The rapid neutron capture process creates about half of all elements heavier than iron. It requires extremely neutron-rich environments where capture occurs faster than beta decay. Confirmed sites include neutron star mergers (as observed in GW170817/GRB170817A). Core-collapse supernovae may also contribute, particularly through neutrino-driven winds, but their r-process yield is debated.", "aliases": "r-process|neutron star merger"},
            {"question": "Explain the tension in measurements of the Hubble constant (H₀ tension).", "answer": "Local measurements (e.g., Cepheid-calibrated Type Ia supernovae by SH0ES) give H₀ ≈ 73 km/s/Mpc, while CMB-based measurements (Planck, assuming ΛCDM) give H₀ ≈ 67.4 km/s/Mpc. The ~5σ discrepancy might indicate new physics (early dark energy, modified neutrino physics), systematic errors in local distance ladders, or issues with the standard cosmological model.", "aliases": "Hubble tension"},
        ],
        5: [
            {"question": "Discuss the current state of primordial gravitational wave detection and its implications for inflationary cosmology.", "answer": "Inflationary models predict a background of primordial gravitational waves (tensor perturbations) that would produce B-mode polarisation in the CMB. The tensor-to-scalar ratio r constrains the energy scale of inflation. BICEP/Keck currently places r < 0.036 (95% CL). Many large-field inflation models predict detectable r, while small-field models can have arbitrarily small r. Future experiments (CMB-S4, LiteBIRD) aim to reach r ~ 0.001, potentially ruling out entire classes of models.", "aliases": ""},
            {"question": "What is the current understanding of fast radio bursts (FRBs) and their potential origins?", "answer": "FRBs are millisecond-duration radio transients with high dispersion measures indicating extragalactic origin. The discovery of FRB 20200428 from galactic magnetar SGR 1935+2154 confirmed magnetars as at least one source. Repeating vs non-repeating FRBs may have different mechanisms. Proposed models include magnetar flares, neutron star mergers, and exotic scenarios. Their dispersion measures probe the intergalactic medium, making them potential cosmological probes.", "aliases": "FRB"},
            {"question": "Assess the current evidence for and against dark matter alternatives like MOND.", "answer": "MOND (Modified Newtonian Dynamics) successfully predicts galaxy rotation curves without dark matter and the baryonic Tully-Fisher relation. However, it struggles with galaxy clusters (still needs some dark matter), the CMB power spectrum (requires relativistic extension TeVeS, which has issues), gravitational lensing (Bullet Cluster), and structure formation. The lack of direct dark matter detection keeps alternatives alive, but the CMB and large-scale structure evidence strongly favours CDM. Hybrid approaches and new frameworks (superfluid dark matter) attempt to capture MOND's galactic successes within a dark matter framework.", "aliases": "MOND"},
            {"question": "What are the prospects and challenges for detecting gravitational waves from the early universe using pulsar timing arrays?", "answer": "Pulsar timing arrays (NANOGrav, EPTA, PPTA) recently reported evidence of a nanohertz gravitational wave background. The primary expected source is supermassive black hole binary inspiral, but contributions from cosmic strings, phase transitions, or primordial gravitational waves are possible. Challenges include: separating the signal from pulsar intrinsic noise, the long observation baselines required (decades), the limited number of stable millisecond pulsars, and distinguishing astrophysical from cosmological sources through spectral shape and anisotropy analysis.", "aliases": "PTA|NANOGrav"},
        ],
    },
}


class ExpertiseGradient(BaseBenchmark):
    benchmark_id = "t31"
    name = "Expertise Gradient"
    pillar = "Knowledge Boundary Awareness"
    primary_metric = "mean_curve_correlation"

    # ---- dataset ----
    def generate_dataset(self) -> list[dict]:
        items = []
        for domain_key, domain_data in _QUESTIONS.items():
            domain_name = domain_data["domain"]
            for level in range(1, 6):
                for q in domain_data[level]:
                    items.append({
                        "domain": domain_name,
                        "domain_key": domain_key,
                        "level": level,
                        "question": q["question"],
                        "reference_answer": q["answer"],
                        "aliases": q.get("aliases", ""),
                    })
        return items  # 3 * 5 * 3 = 45

    # ---- per-item ----
    def run_item(self, item: dict) -> dict:
        prompt = (
            f'Domain: {item["domain"]}\n'
            f'Difficulty level: {item["level"]} (1=introductory, 5=frontier research)\n\n'
            f'Question: {item["question"]}\n\n'
            "Answer the question and provide your confidence (0-100) that "
            "your answer is correct.\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"answer": "your answer", "confidence": <0-100>}'
        )
        resp = self.llm.prompt_json(prompt)

        confidence = resp.get("confidence", 50)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 50.0
        confidence = max(0.0, min(100.0, float(confidence)))

        model_answer = resp.get("answer", "")

        correct = check_answer(
            model_answer,
            item["reference_answer"],
            accept_aliases=item.get("aliases"),
            llm=self.llm,
            question=item["question"],
        )

        return {
            "domain": item["domain"],
            "domain_key": item["domain_key"],
            "level": item["level"],
            "question": item["question"],
            "model_answer": model_answer,
            "reference_answer": item["reference_answer"],
            "correct": correct,
            "confidence": confidence,
        }

    # ---- aggregate ----
    def aggregate(self, results: list[dict]) -> dict:
        # Group by domain
        domains: dict[str, list[dict]] = {}
        for r in results:
            domains.setdefault(r["domain_key"], []).append(r)

        correlations = []
        domain_details = {}

        for domain_key, items in domains.items():
            # Compute accuracy and mean confidence per level
            level_acc = {}
            level_conf = {}
            for level in range(1, 6):
                level_items = [i for i in items if i["level"] == level]
                if level_items:
                    level_acc[level] = np.mean([i["correct"] for i in level_items])
                    level_conf[level] = np.mean([i["confidence"] for i in level_items])

            if len(level_acc) >= 3:
                levels = sorted(level_acc.keys())
                acc_curve = [float(level_acc[l]) for l in levels]
                conf_curve = [float(level_conf[l]) for l in levels]
                r_val = pearson_r(acc_curve, conf_curve)
            else:
                acc_curve = []
                conf_curve = []
                r_val = 0.0

            correlations.append(r_val)
            domain_name = items[0]["domain"] if items else domain_key
            domain_details[domain_name] = {
                "pearson_r": float(r_val),
                "accuracy_curve": acc_curve,
                "confidence_curve": conf_curve,
                "n_items": len(items),
            }

        mean_curve_correlation = float(np.mean(correlations)) if correlations else 0.0

        return {
            "mean_curve_correlation": mean_curve_correlation,
            "per_domain": domain_details,
            "n_domains": len(domains),
            "total_items": len(results),
        }
