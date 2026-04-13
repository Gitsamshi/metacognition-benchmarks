"""T48 — Abstention ROC.

The headline metacognition metric.  A large set of factual questions spanning
multiple domains and difficulty levels.  For each question the model provides
an answer and a confidence score.  We compute AUROC treating confidence as a
binary classifier for correctness — a perfectly calibrated model that is
confident exactly when it is right will score 1.0.
"""

from ..runner import BaseBenchmark
from ..judge import check_answer, llm_judge_correctness, llm_judge_binary
from ..metrics import compute_ece, compute_abstention_auroc, compute_auroc, pearson_r, spearman_rho
from ..llm_client import ConversationSession
import json
import numpy as np


# ---------------------------------------------------------------------------
# Dataset — ~200 questions across domains and difficulty levels
# ---------------------------------------------------------------------------

_QUESTIONS: list[dict] = [
    # ===== SCIENCE (40 questions) =====
    # -- easy --
    {"id": "sci_e01", "domain": "science", "difficulty": "easy", "question": "What planet is known as the Red Planet?", "answer": "Mars"},
    {"id": "sci_e02", "domain": "science", "difficulty": "easy", "question": "What is the chemical formula for water?", "answer": "H2O"},
    {"id": "sci_e03", "domain": "science", "difficulty": "easy", "question": "What force keeps us on the ground?", "answer": "Gravity"},
    {"id": "sci_e04", "domain": "science", "difficulty": "easy", "question": "What organ pumps blood through the body?", "answer": "Heart"},
    {"id": "sci_e05", "domain": "science", "difficulty": "easy", "question": "How many legs does an insect have?", "answer": "6"},
    {"id": "sci_e06", "domain": "science", "difficulty": "easy", "question": "What is the closest star to Earth?", "answer": "The Sun"},
    {"id": "sci_e07", "domain": "science", "difficulty": "easy", "question": "What gas do humans exhale?", "answer": "Carbon dioxide"},
    {"id": "sci_e08", "domain": "science", "difficulty": "easy", "question": "What is the hardest natural substance?", "answer": "Diamond"},
    {"id": "sci_e09", "domain": "science", "difficulty": "easy", "question": "What part of the plant conducts photosynthesis?", "answer": "Leaf|Leaves"},
    {"id": "sci_e10", "domain": "science", "difficulty": "easy", "question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    # -- medium --
    {"id": "sci_m01", "domain": "science", "difficulty": "medium", "question": "What is the powerhouse of the cell?", "answer": "Mitochondria"},
    {"id": "sci_m02", "domain": "science", "difficulty": "medium", "question": "What is the atomic number of carbon?", "answer": "6"},
    {"id": "sci_m03", "domain": "science", "difficulty": "medium", "question": "What type of bond involves the sharing of electrons?", "answer": "Covalent bond"},
    {"id": "sci_m04", "domain": "science", "difficulty": "medium", "question": "What is the SI unit of electric current?", "answer": "Ampere"},
    {"id": "sci_m05", "domain": "science", "difficulty": "medium", "question": "What particle in an atom has a negative charge?", "answer": "Electron"},
    {"id": "sci_m06", "domain": "science", "difficulty": "medium", "question": "What is the pH of pure water?", "answer": "7"},
    {"id": "sci_m07", "domain": "science", "difficulty": "medium", "question": "What law states that energy cannot be created or destroyed?", "answer": "First law of thermodynamics|Law of conservation of energy"},
    {"id": "sci_m08", "domain": "science", "difficulty": "medium", "question": "What is the most abundant gas in Earth's atmosphere?", "answer": "Nitrogen"},
    {"id": "sci_m09", "domain": "science", "difficulty": "medium", "question": "What phenomenon causes the sky to appear blue?", "answer": "Rayleigh scattering"},
    {"id": "sci_m10", "domain": "science", "difficulty": "medium", "question": "What is the speed of sound in air at sea level, approximately in m/s?", "answer": "343"},
    # -- hard --
    {"id": "sci_h01", "domain": "science", "difficulty": "hard", "question": "What is the Chandrasekhar limit in solar masses?", "answer": "1.4"},
    {"id": "sci_h02", "domain": "science", "difficulty": "hard", "question": "What enzyme unwinds the DNA double helix during replication?", "answer": "Helicase"},
    {"id": "sci_h03", "domain": "science", "difficulty": "hard", "question": "What is the name of the hypothetical particle that mediates gravity?", "answer": "Graviton"},
    {"id": "sci_h04", "domain": "science", "difficulty": "hard", "question": "In quantum mechanics, what principle states that you cannot simultaneously know the exact position and momentum of a particle?", "answer": "Heisenberg uncertainty principle"},
    {"id": "sci_h05", "domain": "science", "difficulty": "hard", "question": "What is the Schwarzschild radius formula for a non-rotating black hole?", "answer": "r_s = 2GM/c^2|2GM/c²"},
    {"id": "sci_h06", "domain": "science", "difficulty": "hard", "question": "What is the name of the process by which a heavy nucleus splits into lighter nuclei?", "answer": "Nuclear fission"},
    {"id": "sci_h07", "domain": "science", "difficulty": "hard", "question": "What is the cosmological constant problem in physics?", "answer": "The observed value of the cosmological constant is many orders of magnitude smaller than predicted by quantum field theory"},
    {"id": "sci_h08", "domain": "science", "difficulty": "hard", "question": "What is the Pauli exclusion principle?", "answer": "No two identical fermions can occupy the same quantum state simultaneously"},
    {"id": "sci_h09", "domain": "science", "difficulty": "hard", "question": "What is the critical temperature of superconducting mercury in Kelvin?", "answer": "4.2"},
    {"id": "sci_h10", "domain": "science", "difficulty": "hard", "question": "Name the four fundamental forces of nature.", "answer": "Gravity, electromagnetism, strong nuclear force, weak nuclear force"},
    # -- very hard --
    {"id": "sci_v01", "domain": "science", "difficulty": "very_hard", "question": "What is the fine-structure constant approximately equal to?", "answer": "1/137|0.0073"},
    {"id": "sci_v02", "domain": "science", "difficulty": "very_hard", "question": "What is the Lamb shift?", "answer": "A small difference in energy between two hydrogen energy levels (2S1/2 and 2P1/2) that should be degenerate according to the Dirac equation"},
    {"id": "sci_v03", "domain": "science", "difficulty": "very_hard", "question": "What is the Bekenstein-Hawking entropy formula for a black hole?", "answer": "S = A/(4*l_p^2)|S = kA/4lp²"},
    {"id": "sci_v04", "domain": "science", "difficulty": "very_hard", "question": "What is the mass of the Higgs boson in GeV/c²?", "answer": "125|125.1"},
    {"id": "sci_v05", "domain": "science", "difficulty": "very_hard", "question": "In the Standard Model, how many types of quarks are there and what are their names?", "answer": "6: up, down, charm, strange, top, bottom"},
    {"id": "sci_v06", "domain": "science", "difficulty": "very_hard", "question": "What is the Mpemba effect?", "answer": "Hot water can freeze faster than cold water under certain conditions"},
    {"id": "sci_v07", "domain": "science", "difficulty": "very_hard", "question": "What is the anomalous magnetic moment of the electron (g-2) to the first decimal?", "answer": "0.002319|0.0023"},
    {"id": "sci_v08", "domain": "science", "difficulty": "very_hard", "question": "What is the Casimir effect?", "answer": "An attractive force between two uncharged conducting plates due to quantum vacuum fluctuations"},
    {"id": "sci_v09", "domain": "science", "difficulty": "very_hard", "question": "What is the GIM mechanism in particle physics?", "answer": "A mechanism that explains the suppression of flavor-changing neutral currents, predicting the charm quark"},
    {"id": "sci_v10", "domain": "science", "difficulty": "very_hard", "question": "What is the holographic principle?", "answer": "The information contained in a volume of space can be represented by a theory on the boundary of that region"},

    # ===== HISTORY (40 questions) =====
    # -- easy --
    {"id": "his_e01", "domain": "history", "difficulty": "easy", "question": "In what year did Christopher Columbus first reach the Americas?", "answer": "1492"},
    {"id": "his_e02", "domain": "history", "difficulty": "easy", "question": "Who was the first President of the United States?", "answer": "George Washington"},
    {"id": "his_e03", "domain": "history", "difficulty": "easy", "question": "What ancient civilization built the pyramids at Giza?", "answer": "Ancient Egypt|Egyptians"},
    {"id": "his_e04", "domain": "history", "difficulty": "easy", "question": "What wall divided East and West Berlin?", "answer": "Berlin Wall"},
    {"id": "his_e05", "domain": "history", "difficulty": "easy", "question": "Who wrote the Declaration of Independence?", "answer": "Thomas Jefferson"},
    {"id": "his_e06", "domain": "history", "difficulty": "easy", "question": "What year did World War I begin?", "answer": "1914"},
    {"id": "his_e07", "domain": "history", "difficulty": "easy", "question": "What empire was ruled by Julius Caesar?", "answer": "Roman Empire"},
    {"id": "his_e08", "domain": "history", "difficulty": "easy", "question": "Who was the British Prime Minister during most of World War II?", "answer": "Winston Churchill"},
    {"id": "his_e09", "domain": "history", "difficulty": "easy", "question": "On what date did the Japanese attack Pearl Harbor?", "answer": "December 7, 1941"},
    {"id": "his_e10", "domain": "history", "difficulty": "easy", "question": "What document begins with 'We the People'?", "answer": "United States Constitution"},
    # -- medium --
    {"id": "his_m01", "domain": "history", "difficulty": "medium", "question": "What treaty ended World War I?", "answer": "Treaty of Versailles"},
    {"id": "his_m02", "domain": "history", "difficulty": "medium", "question": "Who led the Mongol Empire at its greatest extent?", "answer": "Genghis Khan"},
    {"id": "his_m03", "domain": "history", "difficulty": "medium", "question": "In what year did the French Revolution begin?", "answer": "1789"},
    {"id": "his_m04", "domain": "history", "difficulty": "medium", "question": "What was the name of the ship on which the Pilgrims sailed to America?", "answer": "Mayflower"},
    {"id": "his_m05", "domain": "history", "difficulty": "medium", "question": "Who was the last Tsar of Russia?", "answer": "Nicholas II"},
    {"id": "his_m06", "domain": "history", "difficulty": "medium", "question": "What event is commonly considered the start of the Protestant Reformation?", "answer": "Martin Luther posting his 95 Theses"},
    {"id": "his_m07", "domain": "history", "difficulty": "medium", "question": "What was the Manhattan Project?", "answer": "The US program to develop the first nuclear weapons during WWII"},
    {"id": "his_m08", "domain": "history", "difficulty": "medium", "question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"id": "his_m09", "domain": "history", "difficulty": "medium", "question": "What year did the Soviet Union dissolve?", "answer": "1991"},
    {"id": "his_m10", "domain": "history", "difficulty": "medium", "question": "What battle is considered the turning point of the American Civil War?", "answer": "Battle of Gettysburg"},
    # -- hard --
    {"id": "his_h01", "domain": "history", "difficulty": "hard", "question": "What was the Sykes-Picot Agreement?", "answer": "A secret 1916 agreement between Britain and France to divide the Ottoman Empire's Middle Eastern territories"},
    {"id": "his_h02", "domain": "history", "difficulty": "hard", "question": "Who was the first woman to win a Nobel Prize and in what year?", "answer": "Marie Curie in 1903"},
    {"id": "his_h03", "domain": "history", "difficulty": "hard", "question": "What was the Defenestration of Prague and when did it occur?", "answer": "An incident in 1618 where Protestant nobles threw Catholic officials out of a window in Prague Castle, triggering the Thirty Years' War"},
    {"id": "his_h04", "domain": "history", "difficulty": "hard", "question": "What was the Dreyfus Affair?", "answer": "A political scandal in France (1894-1906) involving the wrongful conviction of Jewish French officer Alfred Dreyfus for treason"},
    {"id": "his_h05", "domain": "history", "difficulty": "hard", "question": "What year was the Code of Hammurabi created?", "answer": "Around 1754 BC|1754 BCE"},
    {"id": "his_h06", "domain": "history", "difficulty": "hard", "question": "Who was Suleiman the Magnificent?", "answer": "The longest-reigning Sultan of the Ottoman Empire (1520-1566), who presided over its golden age"},
    {"id": "his_h07", "domain": "history", "difficulty": "hard", "question": "What was the Taiping Rebellion?", "answer": "A massive civil war in China (1850-1864) led by Hong Xiuquan against the Qing dynasty"},
    {"id": "his_h08", "domain": "history", "difficulty": "hard", "question": "What was the Congress of Vienna (1814-1815) primarily intended to accomplish?", "answer": "Restore the balance of power in Europe after the Napoleonic Wars"},
    {"id": "his_h09", "domain": "history", "difficulty": "hard", "question": "What was the Zimmermann Telegram?", "answer": "A secret 1917 German diplomatic message proposing a military alliance with Mexico against the United States"},
    {"id": "his_h10", "domain": "history", "difficulty": "hard", "question": "Who was Mansa Musa and why is he historically significant?", "answer": "Emperor of the Mali Empire in the 14th century, considered one of the wealthiest individuals in history"},
    # -- very hard --
    {"id": "his_v01", "domain": "history", "difficulty": "very_hard", "question": "What was the Treaty of Tordesillas (1494) and which countries were involved?", "answer": "An agreement between Spain and Portugal dividing newly discovered lands outside Europe between them"},
    {"id": "his_v02", "domain": "history", "difficulty": "very_hard", "question": "What was the Nika Revolt of 532 AD?", "answer": "A massive riot in Constantinople against Emperor Justinian I, nearly toppling his rule"},
    {"id": "his_v03", "domain": "history", "difficulty": "very_hard", "question": "What was the War of the Roses and which houses were involved?", "answer": "A series of English civil wars (1455-1487) between the House of Lancaster and the House of York"},
    {"id": "his_v04", "domain": "history", "difficulty": "very_hard", "question": "What was the Delian League?", "answer": "An alliance of Greek city-states led by Athens, formed around 478 BC to fight the Persian Empire"},
    {"id": "his_v05", "domain": "history", "difficulty": "very_hard", "question": "Who was Ashoka and what empire did he rule?", "answer": "An Indian emperor of the Maurya dynasty (c. 268-232 BC) who converted to Buddhism after the Kalinga War"},
    {"id": "his_v06", "domain": "history", "difficulty": "very_hard", "question": "What was the Edict of Nantes (1598)?", "answer": "A decree by Henry IV of France granting Huguenots (French Protestants) substantial rights"},
    {"id": "his_v07", "domain": "history", "difficulty": "very_hard", "question": "What was the significance of the Battle of Lepanto (1571)?", "answer": "A decisive naval victory of the Holy League against the Ottoman Empire, ending Ottoman naval dominance in the Mediterranean"},
    {"id": "his_v08", "domain": "history", "difficulty": "very_hard", "question": "What was the Scramble for Africa?", "answer": "The rapid colonization of Africa by European powers during the 1880s-1914, formalized at the Berlin Conference of 1884-85"},
    {"id": "his_v09", "domain": "history", "difficulty": "very_hard", "question": "What was the Investiture Controversy?", "answer": "A conflict between the Pope and Holy Roman Emperor over the appointment of church officials (11th-12th centuries)"},
    {"id": "his_v10", "domain": "history", "difficulty": "very_hard", "question": "What was the Treaty of Westphalia (1648) and why is it significant?", "answer": "Peace treaties ending the Thirty Years' War, establishing the principle of state sovereignty in international relations"},

    # ===== GEOGRAPHY (40 questions) =====
    # -- easy --
    {"id": "geo_e01", "domain": "geography", "difficulty": "easy", "question": "What is the largest continent by area?", "answer": "Asia"},
    {"id": "geo_e02", "domain": "geography", "difficulty": "easy", "question": "What ocean lies between Europe and North America?", "answer": "Atlantic Ocean"},
    {"id": "geo_e03", "domain": "geography", "difficulty": "easy", "question": "What is the longest river in the world?", "answer": "Nile|Amazon"},
    {"id": "geo_e04", "domain": "geography", "difficulty": "easy", "question": "What country has the largest population?", "answer": "India|China"},
    {"id": "geo_e05", "domain": "geography", "difficulty": "easy", "question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"id": "geo_e06", "domain": "geography", "difficulty": "easy", "question": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
    {"id": "geo_e07", "domain": "geography", "difficulty": "easy", "question": "What desert is the largest hot desert on Earth?", "answer": "Sahara"},
    {"id": "geo_e08", "domain": "geography", "difficulty": "easy", "question": "What is the smallest country in the world by area?", "answer": "Vatican City"},
    {"id": "geo_e09", "domain": "geography", "difficulty": "easy", "question": "On which continent is Brazil located?", "answer": "South America"},
    {"id": "geo_e10", "domain": "geography", "difficulty": "easy", "question": "What is the capital of Australia?", "answer": "Canberra"},
    # -- medium --
    {"id": "geo_m01", "domain": "geography", "difficulty": "medium", "question": "What is the deepest point in the ocean?", "answer": "Mariana Trench|Challenger Deep"},
    {"id": "geo_m02", "domain": "geography", "difficulty": "medium", "question": "What is the largest lake by surface area?", "answer": "Caspian Sea"},
    {"id": "geo_m03", "domain": "geography", "difficulty": "medium", "question": "What country has the most time zones?", "answer": "France"},
    {"id": "geo_m04", "domain": "geography", "difficulty": "medium", "question": "What strait separates Europe from Africa?", "answer": "Strait of Gibraltar"},
    {"id": "geo_m05", "domain": "geography", "difficulty": "medium", "question": "What is the capital of New Zealand?", "answer": "Wellington"},
    {"id": "geo_m06", "domain": "geography", "difficulty": "medium", "question": "What river flows through Cairo?", "answer": "Nile"},
    {"id": "geo_m07", "domain": "geography", "difficulty": "medium", "question": "Which African country was formerly known as Abyssinia?", "answer": "Ethiopia"},
    {"id": "geo_m08", "domain": "geography", "difficulty": "medium", "question": "What is the largest island in the world?", "answer": "Greenland"},
    {"id": "geo_m09", "domain": "geography", "difficulty": "medium", "question": "What mountain range separates Europe from Asia?", "answer": "Ural Mountains"},
    {"id": "geo_m10", "domain": "geography", "difficulty": "medium", "question": "What is the driest continent?", "answer": "Antarctica"},
    # -- hard --
    {"id": "geo_h01", "domain": "geography", "difficulty": "hard", "question": "What is the capital of Myanmar?", "answer": "Naypyidaw"},
    {"id": "geo_h02", "domain": "geography", "difficulty": "hard", "question": "What is the highest waterfall in the world?", "answer": "Angel Falls"},
    {"id": "geo_h03", "domain": "geography", "difficulty": "hard", "question": "What country is home to the ancient city of Petra?", "answer": "Jordan"},
    {"id": "geo_h04", "domain": "geography", "difficulty": "hard", "question": "What is the longest mountain range on Earth?", "answer": "Andes"},
    {"id": "geo_h05", "domain": "geography", "difficulty": "hard", "question": "What is the capital of Kazakhstan?", "answer": "Astana"},
    {"id": "geo_h06", "domain": "geography", "difficulty": "hard", "question": "What three countries does the Mekong River flow through before reaching Vietnam?", "answer": "China, Myanmar, Laos|China, Laos, Thailand"},
    {"id": "geo_h07", "domain": "geography", "difficulty": "hard", "question": "What is the only country in the world that spans both the equator and a tropic?", "answer": "Brazil"},
    {"id": "geo_h08", "domain": "geography", "difficulty": "hard", "question": "What sea lies between Italy and Croatia?", "answer": "Adriatic Sea"},
    {"id": "geo_h09", "domain": "geography", "difficulty": "hard", "question": "What is the second-largest French-speaking city in the world by population?", "answer": "Kinshasa"},
    {"id": "geo_h10", "domain": "geography", "difficulty": "hard", "question": "What African country is entirely surrounded by South Africa?", "answer": "Lesotho"},
    # -- very hard --
    {"id": "geo_v01", "domain": "geography", "difficulty": "very_hard", "question": "What is the tripoint where Argentina, Brazil, and Paraguay meet?", "answer": "Triple Frontier|Tríplice Fronteira"},
    {"id": "geo_v02", "domain": "geography", "difficulty": "very_hard", "question": "What is the highest capital city in the world by elevation?", "answer": "La Paz|Quito"},
    {"id": "geo_v03", "domain": "geography", "difficulty": "very_hard", "question": "What is the Darien Gap?", "answer": "A break in the Pan-American Highway in the dense jungle between Panama and Colombia"},
    {"id": "geo_v04", "domain": "geography", "difficulty": "very_hard", "question": "What is the exonym for the country whose endonym is Sakartvelo?", "answer": "Georgia"},
    {"id": "geo_v05", "domain": "geography", "difficulty": "very_hard", "question": "Name the two countries that the Aral Sea borders.", "answer": "Kazakhstan and Uzbekistan"},
    {"id": "geo_v06", "domain": "geography", "difficulty": "very_hard", "question": "What is the only landlocked country in Southeast Asia?", "answer": "Laos"},
    {"id": "geo_v07", "domain": "geography", "difficulty": "very_hard", "question": "What is the deepest lake in the world and where is it?", "answer": "Lake Baikal in Russia|Siberia"},
    {"id": "geo_v08", "domain": "geography", "difficulty": "very_hard", "question": "What country contains the majority of the Atacama Desert?", "answer": "Chile"},
    {"id": "geo_v09", "domain": "geography", "difficulty": "very_hard", "question": "What is the most populous city in Africa?", "answer": "Lagos|Kinshasa"},
    {"id": "geo_v10", "domain": "geography", "difficulty": "very_hard", "question": "What volcanic island in the Indian Ocean is a French overseas department?", "answer": "Reunion|Réunion"},

    # ===== MATH (40 questions) =====
    # -- easy --
    {"id": "mat_e01", "domain": "math", "difficulty": "easy", "question": "What is 15 + 27?", "answer": "42"},
    {"id": "mat_e02", "domain": "math", "difficulty": "easy", "question": "What is 8 * 9?", "answer": "72"},
    {"id": "mat_e03", "domain": "math", "difficulty": "easy", "question": "What is 100 / 4?", "answer": "25"},
    {"id": "mat_e04", "domain": "math", "difficulty": "easy", "question": "What is 2^8?", "answer": "256"},
    {"id": "mat_e05", "domain": "math", "difficulty": "easy", "question": "What is the square root of 81?", "answer": "9"},
    {"id": "mat_e06", "domain": "math", "difficulty": "easy", "question": "What is 50% of 200?", "answer": "100"},
    {"id": "mat_e07", "domain": "math", "difficulty": "easy", "question": "How many degrees are in a right angle?", "answer": "90"},
    {"id": "mat_e08", "domain": "math", "difficulty": "easy", "question": "What is the perimeter of a square with side length 5?", "answer": "20"},
    {"id": "mat_e09", "domain": "math", "difficulty": "easy", "question": "What is 3! (3 factorial)?", "answer": "6"},
    {"id": "mat_e10", "domain": "math", "difficulty": "easy", "question": "What is the sum of angles in a triangle?", "answer": "180"},
    # -- medium --
    {"id": "mat_m01", "domain": "math", "difficulty": "medium", "question": "What is the derivative of x^3?", "answer": "3x^2|3x²"},
    {"id": "mat_m02", "domain": "math", "difficulty": "medium", "question": "What is the integral of 2x dx?", "answer": "x^2 + C|x² + C"},
    {"id": "mat_m03", "domain": "math", "difficulty": "medium", "question": "Solve: 2x + 5 = 17. What is x?", "answer": "6"},
    {"id": "mat_m04", "domain": "math", "difficulty": "medium", "question": "What is log base 2 of 64?", "answer": "6"},
    {"id": "mat_m05", "domain": "math", "difficulty": "medium", "question": "What is the area of a circle with radius 7? (Use pi = 3.14159)", "answer": "153.94|49pi|49π"},
    {"id": "mat_m06", "domain": "math", "difficulty": "medium", "question": "What is the determinant of the matrix [[1,2],[3,4]]?", "answer": "-2"},
    {"id": "mat_m07", "domain": "math", "difficulty": "medium", "question": "What is the 10th term of the Fibonacci sequence (starting 1,1,...)?", "answer": "55"},
    {"id": "mat_m08", "domain": "math", "difficulty": "medium", "question": "What is the GCD of 84 and 36?", "answer": "12"},
    {"id": "mat_m09", "domain": "math", "difficulty": "medium", "question": "How many permutations of 5 distinct objects are there?", "answer": "120"},
    {"id": "mat_m10", "domain": "math", "difficulty": "medium", "question": "What is sin(30 degrees)?", "answer": "0.5|1/2"},
    # -- hard --
    {"id": "mat_h01", "domain": "math", "difficulty": "hard", "question": "What is the sum of the infinite geometric series 1 + 1/2 + 1/4 + 1/8 + ...?", "answer": "2"},
    {"id": "mat_h02", "domain": "math", "difficulty": "hard", "question": "What is the eigenvalue(s) of the identity matrix I (nxn)?", "answer": "1"},
    {"id": "mat_h03", "domain": "math", "difficulty": "hard", "question": "What is Euler's formula relating e, i, and pi?", "answer": "e^(i*pi) + 1 = 0|e^(iπ) + 1 = 0"},
    {"id": "mat_h04", "domain": "math", "difficulty": "hard", "question": "What is the integral of 1/x dx from 1 to e?", "answer": "1"},
    {"id": "mat_h05", "domain": "math", "difficulty": "hard", "question": "What is the Taylor series expansion of e^x around x=0 up to the x^3 term?", "answer": "1 + x + x^2/2 + x^3/6|1 + x + x²/2 + x³/6"},
    {"id": "mat_h06", "domain": "math", "difficulty": "hard", "question": "How many ways can you partition the integer 5?", "answer": "7"},
    {"id": "mat_h07", "domain": "math", "difficulty": "hard", "question": "What is the value of the Riemann zeta function at s=2?", "answer": "pi^2/6|π²/6"},
    {"id": "mat_h08", "domain": "math", "difficulty": "hard", "question": "What is the Cayley-Hamilton theorem?", "answer": "Every square matrix satisfies its own characteristic equation"},
    {"id": "mat_h09", "domain": "math", "difficulty": "hard", "question": "What is the cardinality of the real numbers compared to the natural numbers?", "answer": "Uncountably infinite|aleph-1|continuum"},
    {"id": "mat_h10", "domain": "math", "difficulty": "hard", "question": "What is the Jacobian determinant used for in multivariable calculus?", "answer": "Changing variables in multiple integrals|describes how areas/volumes transform under a mapping"},
    # -- very hard --
    {"id": "mat_v01", "domain": "math", "difficulty": "very_hard", "question": "What is the Banach-Tarski paradox?", "answer": "A solid ball can be decomposed into finitely many pieces and reassembled into two balls identical to the original"},
    {"id": "mat_v02", "domain": "math", "difficulty": "very_hard", "question": "State the Fundamental Theorem of Galois Theory.", "answer": "There is a one-to-one correspondence between subgroups of the Galois group and intermediate fields of a Galois extension"},
    {"id": "mat_v03", "domain": "math", "difficulty": "very_hard", "question": "What is the classification of finite simple groups?", "answer": "Every finite simple group belongs to one of 18 families or is one of 26 sporadic groups"},
    {"id": "mat_v04", "domain": "math", "difficulty": "very_hard", "question": "What is the Poincare conjecture (now theorem)?", "answer": "Every simply connected, closed 3-manifold is homeomorphic to the 3-sphere"},
    {"id": "mat_v05", "domain": "math", "difficulty": "very_hard", "question": "What is the rank of the elliptic curve y^2 = x^3 - x over Q?", "answer": "0"},
    {"id": "mat_v06", "domain": "math", "difficulty": "very_hard", "question": "What is a modular form?", "answer": "A complex analytic function on the upper half-plane satisfying a certain transformation property under the action of the modular group and a growth condition"},
    {"id": "mat_v07", "domain": "math", "difficulty": "very_hard", "question": "What is the Birch and Swinnerton-Dyer conjecture about?", "answer": "It relates the rank of an elliptic curve to the order of vanishing of its L-function at s=1"},
    {"id": "mat_v08", "domain": "math", "difficulty": "very_hard", "question": "What is the Langlands program?", "answer": "A vast web of conjectures connecting number theory (Galois representations) with automorphic forms and representation theory"},
    {"id": "mat_v09", "domain": "math", "difficulty": "very_hard", "question": "What is the Stone-Weierstrass theorem?", "answer": "Every continuous function on a compact interval can be uniformly approximated by polynomials"},
    {"id": "mat_v10", "domain": "math", "difficulty": "very_hard", "question": "What does Godel's second incompleteness theorem state?", "answer": "A consistent formal system that is strong enough to encode basic arithmetic cannot prove its own consistency"},

    # ===== LITERATURE & CULTURE (40 questions) =====
    # -- easy --
    {"id": "lit_e01", "domain": "literature", "difficulty": "easy", "question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen"},
    {"id": "lit_e02", "domain": "literature", "difficulty": "easy", "question": "What is the name of Harry Potter's school?", "answer": "Hogwarts"},
    {"id": "lit_e03", "domain": "literature", "difficulty": "easy", "question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"id": "lit_e04", "domain": "literature", "difficulty": "easy", "question": "What fictional detective lived at 221B Baker Street?", "answer": "Sherlock Holmes"},
    {"id": "lit_e05", "domain": "literature", "difficulty": "easy", "question": "Who wrote 'A Tale of Two Cities'?", "answer": "Charles Dickens"},
    {"id": "lit_e06", "domain": "literature", "difficulty": "easy", "question": "What instrument has 88 keys?", "answer": "Piano"},
    {"id": "lit_e07", "domain": "literature", "difficulty": "easy", "question": "In Greek mythology, who is the king of the gods?", "answer": "Zeus"},
    {"id": "lit_e08", "domain": "literature", "difficulty": "easy", "question": "Who composed the 'Four Seasons'?", "answer": "Antonio Vivaldi"},
    {"id": "lit_e09", "domain": "literature", "difficulty": "easy", "question": "What is the name of the hobbit in 'The Lord of the Rings' who carries the Ring?", "answer": "Frodo|Frodo Baggins"},
    {"id": "lit_e10", "domain": "literature", "difficulty": "easy", "question": "Who wrote '1984'?", "answer": "George Orwell"},
    # -- medium --
    {"id": "lit_m01", "domain": "literature", "difficulty": "medium", "question": "What is the opening line of 'Moby-Dick'?", "answer": "Call me Ishmael"},
    {"id": "lit_m02", "domain": "literature", "difficulty": "medium", "question": "Who wrote 'One Hundred Years of Solitude'?", "answer": "Gabriel Garcia Marquez"},
    {"id": "lit_m03", "domain": "literature", "difficulty": "medium", "question": "What art movement is Salvador Dali associated with?", "answer": "Surrealism"},
    {"id": "lit_m04", "domain": "literature", "difficulty": "medium", "question": "Who composed 'The Rite of Spring'?", "answer": "Igor Stravinsky"},
    {"id": "lit_m05", "domain": "literature", "difficulty": "medium", "question": "What novel features the character Jay Gatsby?", "answer": "The Great Gatsby"},
    {"id": "lit_m06", "domain": "literature", "difficulty": "medium", "question": "Who wrote 'The Divine Comedy'?", "answer": "Dante Alighieri"},
    {"id": "lit_m07", "domain": "literature", "difficulty": "medium", "question": "What Japanese form of poetry has a 5-7-5 syllable structure?", "answer": "Haiku"},
    {"id": "lit_m08", "domain": "literature", "difficulty": "medium", "question": "Who directed the film 'Citizen Kane'?", "answer": "Orson Welles"},
    {"id": "lit_m09", "domain": "literature", "difficulty": "medium", "question": "What novel by Fyodor Dostoevsky features the character Raskolnikov?", "answer": "Crime and Punishment"},
    {"id": "lit_m10", "domain": "literature", "difficulty": "medium", "question": "Who sculpted 'David' (the famous Renaissance sculpture)?", "answer": "Michelangelo"},
    # -- hard --
    {"id": "lit_h01", "domain": "literature", "difficulty": "hard", "question": "What is the name of the epic poem attributed to the ancient Sumerian civilization?", "answer": "Epic of Gilgamesh"},
    {"id": "lit_h02", "domain": "literature", "difficulty": "hard", "question": "Who wrote 'Infinite Jest'?", "answer": "David Foster Wallace"},
    {"id": "lit_h03", "domain": "literature", "difficulty": "hard", "question": "What literary movement did the Beats belong to?", "answer": "Beat Generation"},
    {"id": "lit_h04", "domain": "literature", "difficulty": "hard", "question": "Who composed the opera 'Tristan und Isolde'?", "answer": "Richard Wagner"},
    {"id": "lit_h05", "domain": "literature", "difficulty": "hard", "question": "What novel by James Joyce is known for its stream-of-consciousness style and takes place on June 16, 1904?", "answer": "Ulysses"},
    {"id": "lit_h06", "domain": "literature", "difficulty": "hard", "question": "Who wrote 'Things Fall Apart'?", "answer": "Chinua Achebe"},
    {"id": "lit_h07", "domain": "literature", "difficulty": "hard", "question": "What is the Rosetta Stone and why is it important?", "answer": "A stone slab with text in three scripts that enabled the decipherment of Egyptian hieroglyphs"},
    {"id": "lit_h08", "domain": "literature", "difficulty": "hard", "question": "Who is the author of 'The Master and Margarita'?", "answer": "Mikhail Bulgakov"},
    {"id": "lit_h09", "domain": "literature", "difficulty": "hard", "question": "What painting technique uses tiny dots of color to form an image?", "answer": "Pointillism"},
    {"id": "lit_h10", "domain": "literature", "difficulty": "hard", "question": "Who wrote the epic poem 'Paradise Lost'?", "answer": "John Milton"},
    # -- very hard --
    {"id": "lit_v01", "domain": "literature", "difficulty": "very_hard", "question": "What is Oulipo?", "answer": "A French literary group that creates works using constrained writing techniques"},
    {"id": "lit_v02", "domain": "literature", "difficulty": "very_hard", "question": "Who wrote 'The Muqaddimah'?", "answer": "Ibn Khaldun"},
    {"id": "lit_v03", "domain": "literature", "difficulty": "very_hard", "question": "What is the 'Gesamtkunstwerk' concept associated with Wagner?", "answer": "A total work of art that synthesizes all arts (music, drama, visual art) into a unified whole"},
    {"id": "lit_v04", "domain": "literature", "difficulty": "very_hard", "question": "Who wrote 'Hopscotch' (Rayuela)?", "answer": "Julio Cortazar"},
    {"id": "lit_v05", "domain": "literature", "difficulty": "very_hard", "question": "What is the Voynich manuscript?", "answer": "A mysterious illustrated manuscript written in an undeciphered script, dated to the early 15th century"},
    {"id": "lit_v06", "domain": "literature", "difficulty": "very_hard", "question": "Who composed 'Le Sacre du Printemps' and what was notable about its premiere?", "answer": "Igor Stravinsky; its 1913 premiere in Paris caused a near-riot due to its revolutionary music and choreography"},
    {"id": "lit_v07", "domain": "literature", "difficulty": "very_hard", "question": "What is a palimpsest?", "answer": "A manuscript page from which text has been scraped or washed off so it can be reused for new writing"},
    {"id": "lit_v08", "domain": "literature", "difficulty": "very_hard", "question": "Who wrote 'The Tale of Genji' and why is it significant?", "answer": "Murasaki Shikibu; it is often considered the world's first novel, written around 1000 AD in Japan"},
    {"id": "lit_v09", "domain": "literature", "difficulty": "very_hard", "question": "What is 'negative capability' as described by John Keats?", "answer": "The capacity to remain in uncertainties, mysteries, and doubts without irritably reaching after fact and reason"},
    {"id": "lit_v10", "domain": "literature", "difficulty": "very_hard", "question": "Who is the author of 'Ficciones'?", "answer": "Jorge Luis Borges"},
]


class AbstentionROC(BaseBenchmark):
    """T48: Abstention ROC.

    The headline metric for metacognition.  A large question set is presented;
    the model answers and provides a confidence score.  AUROC of confidence as
    a predictor of correctness is the primary metric.
    """

    benchmark_id = "t48"
    name = "Abstention ROC"
    pillar = "Metacognitive Control"
    primary_metric = "auroc"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def generate_dataset(self) -> list[dict]:
        return list(_QUESTIONS)

    # ------------------------------------------------------------------
    # Per‑item execution
    # ------------------------------------------------------------------
    def run_item(self, item: dict) -> dict:
        question = item["question"]
        # Support both "answer" and "correct_answer" field names
        correct_answer = item.get("answer", item.get("correct_answer", ""))

        # accept_aliases may be provided as a separate field (Opus datasets)
        # or embedded in the answer with | separators (programmatic datasets)
        accept_aliases = item.get("accept_aliases")
        if accept_aliases is not None:
            # accept_aliases may be a list or a pipe-separated string
            if isinstance(accept_aliases, list):
                accept_aliases = "|".join(str(a) for a in accept_aliases if a)
            primary_answer = correct_answer
            accept_str = accept_aliases if accept_aliases else None
        else:
            # Split aliases embedded in the answer field
            aliases = correct_answer.split("|") if "|" in correct_answer else None
            primary_answer = aliases[0] if aliases else correct_answer
            accept_str = "|".join(aliases) if aliases else None

        prompt = (
            f"Answer the following question. Also rate your confidence "
            f"that your answer is correct on a scale from 0 to 100.\n\n"
            f"Question: {question}\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"answer": "<your answer>", "confidence": <0-100>}}'
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
        model_answer = str(resp.get("answer", ""))
        confidence = float(resp.get("confidence", 50)) / 100.0
        confidence = max(0.0, min(1.0, confidence))

        # Check correctness
        correct = check_answer(
            model_answer,
            primary_answer,
            accept_aliases=accept_str,
            llm=self.llm,
            question=question,
        )

        return {
            "id": item.get("id", ""),
            "domain": item.get("domain", "unknown"),
            "difficulty": item.get("difficulty", "unknown"),
            "question": question,
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "confidence": confidence,
            "correct": correct,
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {self.primary_metric: 0.5}

        correctness = [float(r["correct"]) for r in results]
        confidences = [r["confidence"] for r in results]

        # Primary metric: AUROC
        auroc = compute_auroc(correctness, confidences)

        # Abstention-style AUROC (coverage-accuracy curve)
        abstention = compute_abstention_auroc(correctness, confidences)

        # ECE
        ece = compute_ece(correctness, confidences)

        # Correlation
        rho = spearman_rho(correctness, confidences) if len(results) >= 3 else 0.0

        # Overall accuracy
        overall_accuracy = float(np.mean(correctness))
        mean_confidence = float(np.mean(confidences))

        # Per-domain breakdown
        domain_metrics: dict[str, dict] = {}
        for domain in sorted(set(r["domain"] for r in results)):
            dom_results = [r for r in results if r["domain"] == domain]
            dom_correct = [float(r["correct"]) for r in dom_results]
            dom_conf = [r["confidence"] for r in dom_results]
            domain_metrics[domain] = {
                "accuracy": float(np.mean(dom_correct)),
                "mean_confidence": float(np.mean(dom_conf)),
                "auroc": compute_auroc(dom_correct, dom_conf) if len(dom_results) >= 5 else None,
                "count": len(dom_results),
            }

        # Per-difficulty breakdown
        difficulty_metrics: dict[str, dict] = {}
        for diff in sorted(set(r["difficulty"] for r in results)):
            diff_results = [r for r in results if r["difficulty"] == diff]
            diff_correct = [float(r["correct"]) for r in diff_results]
            diff_conf = [r["confidence"] for r in diff_results]
            difficulty_metrics[diff] = {
                "accuracy": float(np.mean(diff_correct)),
                "mean_confidence": float(np.mean(diff_conf)),
                "count": len(diff_results),
            }

        return {
            "auroc": auroc,
            "abstention_auroc": abstention["raw_auroc"],
            "normalized_abstention_auroc": abstention["normalized_auroc"],
            "ece": ece,
            "spearman_rho": rho,
            "overall_accuracy": overall_accuracy,
            "mean_confidence": mean_confidence,
            "per_domain": domain_metrics,
            "per_difficulty": difficulty_metrics,
            "n_items": len(results),
        }
