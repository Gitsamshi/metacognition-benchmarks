"""Data models for the metacognition benchmark builder."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import CATEGORY_PREFIX


@dataclass
class Item:
    id: str
    category: str
    question: str
    gold_answer: str
    topic_tag: str
    subtype: str = "normal"
    difficulty: str = "medium"
    rationale: str = ""
    allow_idk: bool = False
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(category: str, index: int) -> str:
        prefix = CATEGORY_PREFIX.get(category, "x")
        return f"{prefix}_{index:03d}"


@dataclass
class RefRun:
    item_id: str
    model: str
    mode: str
    answer: str
    stated_conf: float
    is_correct: bool
    reasoning: str = ""


@dataclass
class KillSignal:
    signal_type: str
    triggered: bool
    detail: str = ""


@dataclass
class ScoredItem:
    item: Item
    ref_runs: list[RefRun] = field(default_factory=list)
    kill_signals: list[KillSignal] = field(default_factory=list)
    marginal_ece_spread: float = 0.0

    @property
    def alive(self) -> bool:
        return all(not ks.triggered for ks in self.kill_signals)


@dataclass
class Beam:
    items: list[Item] = field(default_factory=list)
    ref_runs: list[RefRun] = field(default_factory=list)
    ece_spread: float = 0.0
    per_model_ece: dict[str, float] = field(default_factory=dict)

    def clone(self) -> Beam:
        return Beam(
            items=list(self.items),
            ref_runs=list(self.ref_runs),
            ece_spread=self.ece_spread,
            per_model_ece=dict(self.per_model_ece),
        )

    def category_items(self, category: str) -> list[Item]:
        return [it for it in self.items if it.category == category]

    def all_topic_tags(self, category: str) -> set[str]:
        return {it.topic_tag for it in self.category_items(category)}


@dataclass
class SearchState:
    beams: list[Beam] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {
        "items_generated": 0,
        "items_rejected": 0,
        "api_calls": 0,
    })
