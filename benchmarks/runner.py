"""Benchmark runner — orchestrates dataset loading, task execution, and scoring."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import BenchmarkConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


DATASET_VERSION = "2.0"


@dataclass
class BenchmarkResult:
    """Result from running a single benchmark."""
    benchmark_id: str
    benchmark_name: str
    pillar: str
    primary_metric_name: str
    primary_metric_value: float
    all_metrics: dict = field(default_factory=dict)
    per_item_results: list = field(default_factory=list)
    llm_stats: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    dataset_version: str = DATASET_VERSION
    insufficient_signal: bool = False
    insufficient_signal_reason: Optional[str] = None


class BaseBenchmark:
    """Base class for all metacognition benchmarks."""

    # Subclasses must set these
    benchmark_id: str = ""
    name: str = ""
    pillar: str = ""
    primary_metric: str = ""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.llm = LLMClient(config.bedrock)

    def load_dataset(self) -> list[dict]:
        """Load dataset from JSON file. Override for custom loading."""
        path = Path(self.config.dataset_dir) / f"{self.benchmark_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        # Try to generate if no dataset exists
        return self.generate_dataset()

    def generate_dataset(self) -> list[dict]:
        """Generate dataset programmatically. Override in subclasses."""
        raise FileNotFoundError(
            f"No dataset found for {self.benchmark_id}. "
            f"Place a JSON file at datasets/{self.benchmark_id}.json "
            f"or override generate_dataset()."
        )

    def run_item(self, item: dict) -> dict:
        """Run benchmark on a single dataset item. Must be overridden."""
        raise NotImplementedError

    def aggregate(self, results: list[dict]) -> dict:
        """Aggregate per-item results into benchmark-level metrics. Must be overridden."""
        raise NotImplementedError

    # Set to True in subclasses that use ConversationSession or
    # depend on shared state across run_item calls (e.g., multi-turn).
    requires_sequential: bool = False

    def _make_worker(self):
        """Create an independent copy of this benchmark for parallel execution.

        Each worker gets its own LLMClient to avoid shared state.
        """
        worker = self.__class__(self.config)
        return worker

    def run(self) -> BenchmarkResult:
        """Run the full benchmark pipeline."""
        start = time.time()
        n_workers = self.config.parallel_workers
        if self.requires_sequential:
            n_workers = 1
        try:
            dataset = self.load_dataset()
            logger.info(
                "Running %s (%s) on %d items (workers=%d)",
                self.name, self.benchmark_id, len(dataset), n_workers,
            )

            per_item = []
            if n_workers <= 1:
                # Sequential execution
                for i, item in enumerate(dataset):
                    try:
                        result = self.run_item(item)
                        result["_item_index"] = i
                        per_item.append(result)
                    except Exception as e:
                        logger.warning("Error on item %d of %s: %s", i, self.benchmark_id, e)
                        per_item.append({"_item_index": i, "error": str(e)})

                    if self.config.verbose and (i + 1) % 10 == 0:
                        logger.info("  %s: %d/%d items done", self.benchmark_id, i + 1, len(dataset))
            else:
                # Parallel execution
                def _run_one(args):
                    idx, item, worker = args
                    try:
                        result = worker.run_item(item)
                        result["_item_index"] = idx
                        return result
                    except Exception as e:
                        logger.warning("Error on item %d of %s: %s", idx, self.benchmark_id, e)
                        return {"_item_index": idx, "error": str(e)}

                workers = [self._make_worker() for _ in range(n_workers)]
                tasks = [(i, item, workers[i % n_workers]) for i, item in enumerate(dataset)]

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(_run_one, t): t[0] for t in tasks}
                    done_count = 0
                    for future in as_completed(futures):
                        result = future.result()
                        per_item.append(result)
                        done_count += 1
                        if self.config.verbose and done_count % 10 == 0:
                            logger.info("  %s: %d/%d items done", self.benchmark_id, done_count, len(dataset))

                # Sort back by item index
                per_item.sort(key=lambda x: x.get("_item_index", 0))

            metrics = self.aggregate([r for r in per_item if "error" not in r])
            elapsed = time.time() - start

            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                benchmark_name=self.name,
                pillar=self.pillar,
                primary_metric_name=self.primary_metric,
                primary_metric_value=metrics.get(self.primary_metric, 0.0),
                all_metrics=metrics,
                per_item_results=per_item,
                llm_stats=self.llm.stats,
                elapsed_seconds=elapsed,
            )
        except Exception as e:
            logger.error("Benchmark %s failed: %s", self.benchmark_id, e)
            return BenchmarkResult(
                benchmark_id=self.benchmark_id,
                benchmark_name=self.name,
                pillar=self.pillar,
                primary_metric_name=self.primary_metric,
                primary_metric_value=0.0,
                error=str(e),
                elapsed_seconds=time.time() - start,
            )


class BenchmarkSuite:
    """Runs all registered benchmarks and produces a report."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmarks: list[type[BaseBenchmark]] = []

    def register(self, benchmark_cls: type[BaseBenchmark]):
        """Register a benchmark class."""
        self.benchmarks.append(benchmark_cls)

    def run_all(self, filter_ids: Optional[list[str]] = None) -> list[BenchmarkResult]:
        """Run all (or filtered) benchmarks sequentially."""
        results = []
        for cls in self.benchmarks:
            if filter_ids and cls.benchmark_id not in filter_ids:
                continue
            bench = cls(self.config)
            result = bench.run()
            results.append(result)
            logger.info(
                "[%s] %s = %.4f (%.1fs)",
                result.benchmark_id,
                result.primary_metric_name,
                result.primary_metric_value,
                result.elapsed_seconds,
            )
        return results

    def save_results(self, results: list[BenchmarkResult], output_dir: Optional[str] = None):
        """Save results to JSON files."""
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Summary
        summary = []
        for r in results:
            summary.append({
                "benchmark_id": r.benchmark_id,
                "benchmark_name": r.benchmark_name,
                "pillar": r.pillar,
                "primary_metric": r.primary_metric_name,
                "primary_value": r.primary_metric_value,
                "all_metrics": r.all_metrics,
                "elapsed_seconds": r.elapsed_seconds,
                "llm_stats": r.llm_stats,
                "error": r.error,
            })
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # Per-benchmark detail files
        for r in results:
            detail = asdict(r)
            with open(out / f"{r.benchmark_id}_detail.json", "w") as f:
                json.dump(detail, f, indent=2, ensure_ascii=False, default=str)

        # Pillar-level aggregation
        pillar_scores = {}
        for r in results:
            if r.error:
                continue
            pillar_scores.setdefault(r.pillar, []).append(r.primary_metric_value)
        pillar_summary = {
            p: {"mean": float(np.mean(v)), "benchmarks": len(v)}
            for p, v in pillar_scores.items()
        }
        with open(out / "pillar_summary.json", "w") as f:
            json.dump(pillar_summary, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", out)

    @staticmethod
    def print_report(results: list[BenchmarkResult]):
        """Print a formatted report to stdout."""
        print("\n" + "=" * 80)
        print("METACOGNITION BENCHMARK RESULTS")
        print("=" * 80)

        by_pillar: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_pillar.setdefault(r.pillar, []).append(r)

        total_scores = []
        for pillar, bench_results in sorted(by_pillar.items()):
            print(f"\n--- {pillar} ---")
            for r in sorted(bench_results, key=lambda x: x.benchmark_id):
                status = "ERROR" if r.error else f"{r.primary_metric_value:.4f}"
                print(f"  [{r.benchmark_id}] {r.benchmark_name:.<50s} {r.primary_metric_name}: {status}")
                if not r.error:
                    total_scores.append(r.primary_metric_value)

        if total_scores:
            print(f"\n{'=' * 80}")
            print(f"Overall Mean Score: {np.mean(total_scores):.4f} (across {len(total_scores)} benchmarks)")
            print(f"{'=' * 80}\n")
