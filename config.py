"""Experiment settings and deterministic execution controls for OEP-CVR."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    dimension: int = 512
    classes: int = 306
    slots: int = 12
    light_slots: int = 4
    temporal_layers: int = 2
    attention_heads: int = 8
    feedforward_dimension: int = 2048
    dropout: float = 0.10
    alpha: float = 0.10
    pool_size: int = 100
    output_size: int = 10
    review_budget: int = 3
    maximum_interpretations: int = 16
    maximum_conditions: int = 12
    contrastive_temperature: float = 0.07
    classification_weight: float = 1.0
    ordering_weight: float = 0.20
    value_weight: float = 0.10
    incorrect_edge_penalty: float = 2.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_queries: int = 128
    maximum_negatives: int = 32
    temporal_epochs: int = 30
    ordering_epochs: int = 40
    value_epochs: int = 15
    patience: int = 10
    seed: int = 2022

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.dimension % self.attention_heads:
            raise ValueError("The representation dimension must be divisible by attention heads.")
        if not (0 < self.light_slots <= self.slots):
            raise ValueError("The light path requires distinct temporal slots.")
        if self.classes < 2 or not 0 < self.alpha < 1:
            raise ValueError("Invalid class count or conformal miscoverage target.")
        if not 0 < self.output_size <= self.pool_size:
            raise ValueError("The output size must not exceed the fixed candidate pool.")
        if self.review_budget < 0 or self.maximum_interpretations < 1 or self.maximum_conditions < 1:
            raise ValueError("Invalid review or interpretation limits.")
        if self.contrastive_temperature <= 0 or self.batch_queries < 1 or self.maximum_negatives < 1:
            raise ValueError("Invalid optimization settings.")

    @property
    def light_indices(self) -> tuple[int, ...]:
        return tuple(round(i * (self.slots - 1) / max(1, self.light_slots - 1))
                     for i in range(self.light_slots))

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
