"""Dataset generator using Opus 4.6 to create hard benchmark items.

The key principle: datasets must be hard enough that Haiku 4.5 fails on
~40-60% of items. Opus 4.6 generates questions, correct answers, and
metadata. The questions should span a difficulty range from trivially easy
to genuinely obscure/adversarial.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .config import BedrockConfig

logger = logging.getLogger(__name__)

# Opus 4.6 model ID for dataset generation
OPUS_MODEL_ID = "us.anthropic.claude-opus-4-6-v1"

MAX_WORKERS = 16


def get_opus_client(region: str = "us-east-1") -> LLMClient:
    """Create an LLM client configured for Opus 4.6."""
    config = BedrockConfig(
        model_id=OPUS_MODEL_ID,
        region_name=region,
        max_tokens=8192,
        temperature=0.7,  # some creativity for diverse questions
    )
    return LLMClient(config)


def _call_one_batch(args):
    """Worker function for parallel batch generation."""
    region, prompt_text, batch_id = args
    client = get_opus_client(region)
    try:
        resp = client.prompt_json(prompt_text)
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            for key in ("items", "questions", "facts", "statements", "pairs",
                        "scenarios", "problems", "passages", "tasks", "entities"):
                if key in resp and isinstance(resp[key], list):
                    return resp[key]
            for v in resp.values():
                if isinstance(v, list):
                    return v
        logger.warning("Batch %d: could not parse response", batch_id)
        return []
    except Exception as e:
        logger.warning("Batch %d failed: %s", batch_id, e)
        return []


def generate_items_with_opus(
    llm: LLMClient,
    prompt: str,
    n_items: int,
    batch_size: int = 20,
    max_workers: int = MAX_WORKERS,
    region: str = "us-east-1",
) -> list[dict]:
    """Generate dataset items by prompting Opus in parallel batches.

    Args:
        llm: Opus LLM client (used for single-threaded fallback).
        prompt: Prompt template with {n} placeholder for item count.
        n_items: Total number of items to generate.
        batch_size: Items per LLM call.
        max_workers: Number of parallel Opus calls.
        region: AWS region.

    Returns:
        List of dicts (parsed from JSON).
    """
    n_batches = (n_items + batch_size - 1) // batch_size
    tasks = []
    for i in range(n_batches):
        this_batch = min(batch_size, n_items - i * batch_size)
        filled = prompt.replace("{n}", str(this_batch))
        # Add batch-specific diversity instruction
        if i > 0:
            filled += f"\n\nThis is batch {i+1} of {n_batches}. Generate DIFFERENT items from other batches — vary topics, regions, time periods, and difficulty levels."
        tasks.append((region, filled, i))

    all_items = []
    workers = min(max_workers, n_batches)
    logger.info("Generating %d items in %d batches with %d parallel workers", n_items, n_batches, workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_call_one_batch, t): t[2] for t in tasks}
        for future in as_completed(futures):
            batch_id = futures[future]
            items = future.result()
            all_items.extend(items)
            logger.info("Batch %d done: %d items (total: %d)", batch_id, len(items), len(all_items))

    # If we got fewer items than requested, do sequential fill
    retries = 0
    while len(all_items) < n_items and retries < 3:
        deficit = n_items - len(all_items)
        logger.info("Filling deficit of %d items sequentially...", deficit)
        filled = prompt.replace("{n}", str(min(deficit, batch_size)))
        filled += "\n\nGenerate items that are DIFFERENT from all previous batches."
        result = _call_one_batch((region, filled, 999))
        all_items.extend(result)
        retries += 1

    return all_items[:n_items]


def save_dataset(items: list[dict], benchmark_id: str, dataset_dir: str = "benchmarks/datasets"):
    """Save dataset to JSON file with version metadata."""
    path = Path(dataset_dir) / f"{benchmark_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d items to %s", len(items), path)
