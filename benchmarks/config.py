"""Configuration for the metacognition benchmark suite."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BedrockConfig:
    """Configuration for AWS Bedrock LLM calls."""
    model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    region_name: str = "us-east-1"
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class BenchmarkConfig:
    """Global benchmark configuration."""
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    judge_model_id: Optional[str] = None  # defaults to same as bedrock.model_id
    max_concurrent: int = 5
    parallel_workers: int = 1  # number of parallel workers per benchmark
    output_dir: str = "results"
    dataset_dir: str = "benchmarks/datasets"
    verbose: bool = False
    seed: int = 42

    @property
    def judge_model(self) -> str:
        return self.judge_model_id or self.bedrock.model_id
