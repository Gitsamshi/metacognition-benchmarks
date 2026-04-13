#!/usr/bin/env python3
"""Main entry point for the Metacognition Benchmark Suite.

Usage:
    # Run all benchmarks
    python main.py

    # Run specific benchmarks by ID
    python main.py --benchmarks t01 t11 t48

    # Run a specific pillar
    python main.py --pillar "Confidence Calibration"

    # Configure model
    python main.py --model-id us.anthropic.claude-sonnet-4-20250514 --region us-east-1

    # Generate datasets only (no LLM calls)
    python main.py --generate-datasets

    # Verbose output
    python main.py --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmarks.config import BenchmarkConfig, BedrockConfig
from benchmarks.runner import BenchmarkSuite
from benchmarks.tasks import ALL_BENCHMARKS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Metacognition Benchmark Suite — evaluate LLM self-awareness"
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Benchmark IDs to run (e.g., t01 t11 t48). Default: all."
    )
    parser.add_argument(
        "--pillar", type=str, default=None,
        help="Run only benchmarks from this pillar."
    )
    parser.add_argument(
        "--model-id", type=str, default="us.anthropic.claude-sonnet-4-20250514",
        help="Bedrock model ID."
    )
    parser.add_argument(
        "--judge-model-id", type=str, default=None,
        help="Model ID for LLM-as-judge (defaults to same as --model-id)."
    )
    parser.add_argument(
        "--region", type=str, default="us-east-1",
        help="AWS region for Bedrock."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM temperature."
    )
    parser.add_argument(
        "--parallel-workers", type=int, default=1,
        help="Number of parallel workers per benchmark (default: 1, sequential)."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Max tokens per LLM response."
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output files."
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="benchmarks/datasets",
        help="Directory for dataset files."
    )
    parser.add_argument(
        "--generate-datasets", action="store_true",
        help="Generate sample datasets and exit (no LLM calls)."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available benchmarks and exit."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging."
    )
    return parser.parse_args()


def list_benchmarks():
    """Print all available benchmarks."""
    print("\nAvailable Metacognition Benchmarks:")
    print("=" * 80)

    by_pillar: dict[str, list] = {}
    for cls in ALL_BENCHMARKS:
        by_pillar.setdefault(cls.pillar, []).append(cls)

    for pillar, benchmarks in sorted(by_pillar.items()):
        print(f"\n--- {pillar} ({len(benchmarks)} benchmarks) ---")
        for cls in sorted(benchmarks, key=lambda c: c.benchmark_id):
            print(f"  [{cls.benchmark_id}] {cls.name} (metric: {cls.primary_metric})")

    print(f"\nTotal: {len(ALL_BENCHMARKS)} benchmarks")


def generate_all_datasets(config: BenchmarkConfig):
    """Generate sample datasets for all benchmarks."""
    dataset_dir = Path(config.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for cls in ALL_BENCHMARKS:
        bench = cls(config)
        try:
            dataset = bench.generate_dataset()
            path = dataset_dir / f"{cls.benchmark_id}.json"
            with open(path, "w") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"  [{cls.benchmark_id}] Generated {len(dataset)} items -> {path}")
        except Exception as e:
            print(f"  [{cls.benchmark_id}] FAILED: {e}")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list:
        list_benchmarks()
        return

    config = BenchmarkConfig(
        bedrock=BedrockConfig(
            model_id=args.model_id,
            region_name=args.region,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
        judge_model_id=args.judge_model_id,
        parallel_workers=args.parallel_workers,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        verbose=args.verbose,
    )

    if args.generate_datasets:
        print("Generating sample datasets...")
        generate_all_datasets(config)
        print("\nDone. Datasets saved to:", config.dataset_dir)
        return

    # Build suite
    suite = BenchmarkSuite(config)
    for cls in ALL_BENCHMARKS:
        suite.register(cls)

    # Determine which benchmarks to run
    filter_ids = None
    if args.benchmarks:
        filter_ids = args.benchmarks
    elif args.pillar:
        filter_ids = [
            cls.benchmark_id for cls in ALL_BENCHMARKS
            if cls.pillar == args.pillar
        ]

    # Run
    results = suite.run_all(filter_ids=filter_ids)

    # Save first (in case print_report fails), then report
    suite.save_results(results)
    suite.print_report(results)


if __name__ == "__main__":
    main()
