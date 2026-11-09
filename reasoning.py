"""Joint-budget conformal calibration, evidence ordering, losses, and retrieval statistics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from semantics import Bounds


@dataclass(frozen=True)
class ClassSet:
    labels: frozenset[str]
    fallback: bool = False


class State(str, Enum):
    SUPPORT = "support"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class ConformalCalibrator:
    """Maximum nonconformity across budgets with finite-sample split quantiles."""

    def __init__(self, vocabulary: Sequence[str], alpha: float = 0.10):
        if not 0 < alpha < 1 or len(vocabulary) < 2:
            raise ValueError("Invalid calibration settings.")
        self.vocabulary = tuple(vocabulary)
        self.alpha = alpha
        self.threshold: float | None = None

    def fit(self, budget_probabilities: Mapping[int, np.ndarray], labels: Sequence[int]) -> float:
        if set(budget_probabilities) != {4, 12}:
            raise ValueError("Calibration requires both four- and twelve-slot predictions.")
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or labels.size == 0 or np.any(labels < 0) or np.any(labels >= len(self.vocabulary)):
            raise ValueError("Invalid calibration labels.")
        scores = []
        for budget in (4, 12):
            probs = np.asarray(budget_probabilities[budget], dtype=np.float64)
            if probs.shape != (len(labels), len(self.vocabulary)):
                raise ValueError("Calibration probabilities have an incompatible shape.")
            if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(1), 1, atol=1e-4):
                raise ValueError("Calibration inputs must be normalized probabilities.")
            scores.append(1.0 - probs[np.arange(len(labels)), labels])
        nonconformity = np.maximum(scores[0], scores[1])
        position = math.ceil((len(labels) + 1) * (1 - self.alpha))
        self.threshold = (float(np.partition(nonconformity, position - 1)[position - 1])
                          if position <= len(labels) else float("inf"))
        return self.threshold

    def predict(self, probabilities: Mapping[int, np.ndarray]) -> ClassSet:
        if self.threshold is None:
            raise RuntimeError("Fit the calibrator before obtaining class sets.")
        selections = []
        for budget, probs in probabilities.items():
            if budget not in {4, 12}:
                raise ValueError("Only calibrated frame budgets are valid.")
            array = np.asarray(probs, dtype=np.float64)
            if array.shape != (len(self.vocabulary),) or not np.isfinite(array).all():
                return ClassSet(frozenset(self.vocabulary), True)
            if (array < 0).any() or (array > 1).any() or not np.isclose(array.sum(), 1, atol=1e-4):
                return ClassSet(frozenset(self.vocabulary), True)
            selections.append({self.vocabulary[i] for i in np.flatnonzero(1 - array <= self.threshold)})
        if not selections:
            return ClassSet(frozenset(self.vocabulary), True)
        intersection = set.intersection(*selections)
        return ClassSet(frozenset(intersection), False) if intersection else ClassSet(
            frozenset(self.vocabulary), True)


def states(bounds: Bounds, candidates: Mapping[str, ClassSet]) -> dict[str, State]:
    result = {}
    for video_id, prediction in candidates.items():
        if bounds.fallback or prediction.fallback or not prediction.labels:
            result[video_id] = State.UNKNOWN
        elif prediction.labels.issubset(bounds.lower):
            result[video_id] = State.SUPPORT
        elif prediction.labels.isdisjoint(bounds.upper):
            result[video_id] = State.CONFLICT
        else:
            result[video_id] = State.UNKNOWN
    return result


def edges(evidence: Mapping[str, State]) -> frozenset[tuple[str, str]]:
    supporting = [name for name, state in evidence.items() if state == State.SUPPORT]
    conflicting = [name for name, state in evidence.items() if state == State.CONFLICT]
    return frozenset((left, right) for left in supporting for right in conflicting if left != right)


def ranked_order(scores: Mapping[str, float], evidence: Mapping[str, State],
                 initial_order: Sequence[str]) -> list[str]:
    """Greedy maximum-score topological ordering with stable rank/ID tie resolution."""
    items = tuple(scores)
    if set(items) != set(evidence) or set(items) != set(initial_order):
        raise ValueError("Ranking arguments must describe the same candidate pool.")
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Similarity scores must be finite.")
    if all(item == State.UNKNOWN for item in evidence.values()):
        return list(initial_order)
    first_rank = {name: idx for idx, name in enumerate(initial_order)}
    current_edges = edges(evidence)
    dependencies = {name: set() for name in items}
    successors = {name: set() for name in items}
    for left, right in current_edges:
        dependencies[right].add(left)
        successors[left].add(right)
    output, remaining = [], set(items)
    while remaining:
        ready = [name for name in remaining if not dependencies[name]]
        if not ready:
            raise RuntimeError("The confirmed precedence relations contain a cycle.")
        winner = min(ready, key=lambda name: (-scores[name], first_rank[name], name))
        output.append(winner)
        remaining.remove(winner)
        for right in successors[winner]:
            dependencies[right].discard(winner)
    return output


def multi_positive_loss(scores: torch.Tensor, positive_mask: torch.Tensor,
                        temperature: float = 0.07) -> torch.Tensor:
    if scores.ndim != 1 or positive_mask.shape != scores.shape or not positive_mask.any():
        raise ValueError("A query requires at least one known positive candidate.")
    scaled = scores.float() / temperature
    return -(scaled[positive_mask] - torch.logsumexp(scaled, dim=0)).mean()


def partial_order_loss(scores: torch.Tensor, ids: Sequence[str], evidence: Mapping[str, State],
                       positives: frozenset[str], temperature: float = 0.07) -> torch.Tensor:
    if scores.numel() != len(ids):
        raise ValueError("The score count must equal the number of candidate IDs.")
    position = {name: i for i, name in enumerate(ids)}
    legal = [(left, right) for left, right in edges(evidence)
             if left in positives and right not in positives]
    if not legal:
        return scores.sum() * 0
    return F.softplus(torch.stack([(scores[position[right]] - scores[position[left]])
                                   / temperature for left, right in legal])).mean()


def average_precision_at_k(order: Sequence[str], positives: frozenset[str], k: int = 50) -> float:
    if not positives or k < 1:
        raise ValueError("Average precision requires positive labels and a valid cutoff.")
    correct, area = 0, 0.0
    for rank, video_id in enumerate(order[:k], 1):
        if video_id in positives:
            correct += 1
            area += correct / rank
    return area / min(k, len(positives))


def retrieval_metrics(orders: Sequence[Sequence[str]], positives: Sequence[frozenset[str]]) -> dict[str, float]:
    if not orders or len(orders) != len(positives):
        raise ValueError("A nonempty, paired evaluation set is required.")
    result = {f"mAP@{k}": 100 * float(np.mean([
        average_precision_at_k(order, gold, k) for order, gold in zip(orders, positives)]))
        for k in (5, 10, 25, 50)}
    result["Hit@10"] = 100 * float(np.mean([bool(set(order[:10]) & gold)
                                              for order, gold in zip(orders, positives)]))
    result["Recall@10"] = 100 * float(np.mean([
        len(set(order[:10]) & gold) / len(gold) for order, gold in zip(orders, positives)]))
    return result


def paired_cluster_bootstrap(first: np.ndarray, second: np.ndarray,
                             groups: Sequence[str], iterations: int = 10000,
                             seed: int = 2022) -> dict[str, float | list[float]]:
    """Paired source-video resampling for the difference in query-level AP values."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 1 or first.shape != second.shape or len(groups) != len(first) or not len(first):
        raise ValueError("Paired observations and cluster identifiers must align.")
    unique = list(dict.fromkeys(groups))
    members = [np.flatnonzero(np.asarray(groups) == g) for g in unique]
    rng = np.random.default_rng(seed)
    delta = first - second
    observed = 100 * float(delta.mean())
    draws = np.empty(iterations)
    for index in range(iterations):
        sampled = rng.integers(0, len(unique), len(unique))
        positions = np.concatenate([members[i] for i in sampled])
        draws[index] = 100 * float(delta[positions].mean())
    lower, upper = np.quantile(draws, [0.025, 0.975]).tolist()
    centered = draws - observed
    p = min(1.0, 2 * min(float(np.mean(centered >= observed)),
                         float(np.mean(centered <= observed))))
    return {"delta_pp": observed, "confidence_interval_pp": [lower, upper], "p_two_sided": p}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    output = [0.0] * len(p_values)
    cumulative = 0.0
    for rank, index in enumerate(order):
        cumulative = max(cumulative, min(1.0, (len(p_values) - rank) * float(p_values[index])))
        output[int(index)] = cumulative
    return output
