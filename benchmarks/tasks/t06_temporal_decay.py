"""T06 — Temporal Knowledge Decay.

Questions from different historical time periods. Tests whether the model's
confidence properly tracks its accuracy decay on more recent or very old events
(where training data may be sparser or cut off).

Time buckets: distant (before 1500), middle (1500-1900), recent (1900-2020),
edge (2020-2025, near or beyond training cutoff).

Primary metric: Spearman correlation between time-bucket accuracy and
time-bucket mean confidence (curve_correlation).
"""

from ..runner import BaseBenchmark
from ..judge import check_answer
from ..metrics import compute_ece, spearman_rho, pearson_r

import numpy as np


TIME_BUCKETS = ["distant", "middle", "recent", "edge"]


class TemporalKnowledgeDecay(BaseBenchmark):
    benchmark_id = "t06"
    name = "Temporal Knowledge Decay"
    pillar = "Confidence Calibration"
    primary_metric = "curve_correlation"

    # ------------------------------------------------------------------ #
    #  Dataset generation
    # ------------------------------------------------------------------ #
    def generate_dataset(self) -> list[dict]:
        """~40 questions across 4 time buckets (~10 each)."""
        items: list[dict] = []

        # distant: before 1500
        distant = [
            ("What year was the Great Pyramid of Giza completed approximately?", "2560 BC", "2560|2560 bc|2580", "distant"),
            ("Who was the first emperor of Rome?", "Augustus", "augustus|octavian", "distant"),
            ("What year did the Western Roman Empire fall?", "476 AD", "476", "distant"),
            ("In what century was the Magna Carta signed?", "13th century", "13th|1215", "distant"),
            ("What empire did Genghis Khan found?", "Mongol Empire", "mongol", "distant"),
            ("What ancient Greek city-state was known for its military prowess?", "Sparta", "sparta", "distant"),
            ("Who was the prophet of Islam?", "Muhammad", "muhammad|mohammed", "distant"),
            ("What year did the Black Death reach Europe?", "1347", "1347|1348", "distant"),
            ("Who wrote 'The Art of War'?", "Sun Tzu", "sun tzu", "distant"),
            ("What dynasty built the Great Wall of China in its most recognized form?", "Ming Dynasty", "ming", "distant"),
            ("What ancient civilization built Machu Picchu?", "Inca", "inca|incan", "distant"),
        ]

        # middle: 1500-1900
        middle = [
            ("In what year did Columbus reach the Americas?", "1492", "1492", "middle"),
            ("Who was the monarch during England's defeat of the Spanish Armada?", "Elizabeth I", "elizabeth i|elizabeth", "middle"),
            ("What year was the Treaty of Westphalia signed?", "1648", "1648", "middle"),
            ("Who wrote 'Principia Mathematica' in 1687?", "Isaac Newton", "newton", "middle"),
            ("What year did the French Revolution begin?", "1789", "1789", "middle"),
            ("Who was the first president of the United States?", "George Washington", "washington", "middle"),
            ("What year was the Declaration of Independence signed?", "1776", "1776", "middle"),
            ("When did Napoleon crown himself Emperor of France?", "1804", "1804", "middle"),
            ("What year was the Emancipation Proclamation issued?", "1863", "1863", "middle"),
            ("Who invented the telephone in 1876?", "Alexander Graham Bell", "bell|graham bell", "middle"),
            ("What year was the Suez Canal opened?", "1869", "1869", "middle"),
        ]

        # recent: 1900-2020
        recent = [
            ("What year did World War I begin?", "1914", "1914", "recent"),
            ("Who was the president of the US during most of World War II?", "Franklin D. Roosevelt", "roosevelt|fdr", "recent"),
            ("What year was the United Nations founded?", "1945", "1945", "recent"),
            ("In what year did the Soviet Union launch Sputnik?", "1957", "1957", "recent"),
            ("What year did the Berlin Wall fall?", "1989", "1989", "recent"),
            ("Who was the first person to walk on the Moon?", "Neil Armstrong", "armstrong", "recent"),
            ("What year did the World Wide Web become publicly available?", "1991", "1991", "recent"),
            ("What year did the September 11 attacks occur?", "2001", "2001", "recent"),
            ("Who was elected US president in 2008?", "Barack Obama", "obama", "recent"),
            ("What year was the Paris Climate Agreement signed?", "2015", "2015|2016", "recent"),
            ("What organization declared COVID-19 a pandemic?", "WHO", "who|world health organization", "recent"),
        ]

        # edge: 2020-2025 (near or beyond training cutoff)
        edge = [
            ("What year did the COVID-19 pandemic begin?", "2020", "2019|2020", "edge"),
            ("Who won the 2020 US presidential election?", "Joe Biden", "biden", "edge"),
            ("What major social media platform was rebranded to X in 2023?", "Twitter", "twitter", "edge"),
            ("What AI chatbot was released by OpenAI in November 2022?", "ChatGPT", "chatgpt|chat gpt", "edge"),
            ("What country hosted the 2022 FIFA World Cup?", "Qatar", "qatar", "edge"),
            ("Who won the 2022 FIFA World Cup?", "Argentina", "argentina", "edge"),
            ("What conflict began in February 2022 involving Russia?", "Russia-Ukraine war", "ukraine|russia-ukraine", "edge"),
            ("What was the name of the James Webb Space Telescope's first deep field image year?", "2022", "2022", "edge"),
            ("What AI company created the GPT-4 model?", "OpenAI", "openai", "edge"),
            ("What country joined NATO in 2023?", "Finland", "finland", "edge"),
        ]

        for q, a, aliases, bucket in distant + middle + recent + edge:
            items.append({
                "question": q,
                "answer": a,
                "accept_aliases": aliases,
                "time_bucket": bucket,
            })

        return items

    # ------------------------------------------------------------------ #
    #  Per-item execution
    # ------------------------------------------------------------------ #
    def run_item(self, item: dict) -> dict:
        prompt = (
            f"Answer the following question and state your confidence.\n\n"
            f"Question: {item['question']}\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"answer": "<your answer>", "confidence": <integer 0-100>}}'
        )
        try:
            resp = self.llm.prompt_json(prompt)
        except (ValueError, KeyError):
            raw = self.llm.prompt(
                prompt + "\nRespond with ONLY valid JSON, nothing else."
            )
            try:
                resp = self.llm._extract_json(raw)
            except (ValueError, KeyError):
                resp = {}
        model_answer = str(resp.get("answer", ""))
        confidence = float(resp.get("confidence", 50))
        confidence = max(0.0, min(100.0, confidence))

        # Support both "answer" and "correct_answer" field names
        correct_answer = item.get("answer", item.get("correct_answer", ""))

        # accept_aliases may be a list or a pipe-separated string
        aliases = item.get("accept_aliases")
        if isinstance(aliases, list):
            aliases = "|".join(str(a) for a in aliases if a)

        correct = check_answer(
            model_answer,
            correct_answer,
            accept_aliases=aliases if aliases else None,
            llm=self.llm,
            question=item["question"],
        )

        return {
            "question": item["question"],
            "time_bucket": item.get("time_bucket", "unknown"),
            "model_answer": model_answer,
            "correct_answer": correct_answer,
            "correct": correct,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ #
    #  Aggregation
    # ------------------------------------------------------------------ #
    def aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {"curve_correlation": 0.0}

        # Per-bucket accuracy and mean confidence
        bucket_correct: dict[str, list] = {}
        bucket_conf: dict[str, list] = {}
        for r in results:
            b = r["time_bucket"]
            bucket_correct.setdefault(b, []).append(1.0 if r["correct"] else 0.0)
            bucket_conf.setdefault(b, []).append(r["confidence"] / 100.0)

        bucket_accuracy = {}
        bucket_mean_conf = {}
        per_bucket = {}
        for b in TIME_BUCKETS:
            if b not in bucket_correct:
                continue
            acc = sum(bucket_correct[b]) / len(bucket_correct[b])
            mc = sum(bucket_conf[b]) / len(bucket_conf[b])
            bucket_accuracy[b] = acc
            bucket_mean_conf[b] = mc
            per_bucket[b] = {
                "n": len(bucket_correct[b]),
                "accuracy": acc,
                "mean_confidence": mc,
                "ece": compute_ece(bucket_correct[b], bucket_conf[b]),
            }

        # Curve correlation: do bucket-level accuracies and confidences correlate?
        if len(bucket_accuracy) >= 3:
            ordered_buckets = [b for b in TIME_BUCKETS if b in bucket_accuracy]
            accs = [bucket_accuracy[b] for b in ordered_buckets]
            confs = [bucket_mean_conf[b] for b in ordered_buckets]
            rho = spearman_rho(accs, confs)
            r = pearson_r(accs, confs)
        else:
            rho = 0.0
            r = 0.0

        # Overall metrics
        all_correct = [1.0 if r_["correct"] else 0.0 for r_ in results]
        all_conf = [r_["confidence"] / 100.0 for r_ in results]

        # Confidence drop: does confidence decrease for edge vs recent?
        conf_drop = 0.0
        if "recent" in bucket_mean_conf and "edge" in bucket_mean_conf:
            conf_drop = bucket_mean_conf["recent"] - bucket_mean_conf["edge"]

        # Accuracy drop
        acc_drop = 0.0
        if "recent" in bucket_accuracy and "edge" in bucket_accuracy:
            acc_drop = bucket_accuracy["recent"] - bucket_accuracy["edge"]

        return {
            "curve_correlation": rho,
            "pearson_r_acc_conf": r,
            "edge_confidence_drop": conf_drop,
            "edge_accuracy_drop": acc_drop,
            "per_bucket": per_bucket,
            "ece": compute_ece(all_correct, all_conf),
            "n_items": len(results),
            "overall_accuracy": sum(all_correct) / len(all_correct),
        }
