"""Cost-aware reference/candidate review with global and local evidence refresh."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import torch
from torch import nn

from config import Config
from data import Corpus, Query
from model import OEPModel
from reasoning import (ClassSet, ConformalCalibrator, State, edges, ranked_order,
                       states)
from semantics import Bounds, Grammar, Ontology


class ValueNetwork(nn.Module):
    """Predicts the signed relational gain from twelve decision features."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != 12:
            raise ValueError("Review decisions require exactly twelve features.")
        return self.layers(features.float()).squeeze(-1)


@dataclass(frozen=True)
class Action:
    scope: str
    video_id: str
    cost: float


@dataclass
class PoolState:
    pool: tuple[str, ...]
    initial: tuple[str, ...]
    full_initial: tuple[str, ...]
    scores: dict[str, float]
    embeddings: dict[str, torch.Tensor]
    probabilities: dict[str, dict[int, np.ndarray]]
    reference_embedding: torch.Tensor
    reference_probabilities: dict[int, np.ndarray]
    reviewed: frozenset[str]
    bounds: Bounds
    evidence: dict[str, State]
    order: list[str]


class InferenceEngine:
    """Fixed-pool retrieval and review following the OEP-CVR inference algorithm."""

    def __init__(self, config: Config, model: OEPModel, ontology: Ontology,
                 grammar: Grammar, calibrator: ConformalCalibrator,
                 policy: ValueNetwork | None = None, device: str = "cpu",
                 reference_cost: float = 1.0, candidate_cost: float = 1.0):
        if set(calibrator.vocabulary) != set(ontology.ids) or len(ontology.ids) != config.classes:
            raise ValueError("Model, calibration, and ontology vocabularies must match.")
        if reference_cost <= 0 or candidate_cost <= 0:
            raise ValueError("Review costs must be positive.")
        self.config, self.model, self.ontology, self.grammar = config, model, ontology, grammar
        self.calibrator, self.policy = calibrator, policy
        self.device = torch.device(device)
        self.reference_cost, self.candidate_cost = reference_cost, candidate_cost
        self.model.to(self.device).eval()
        if self.policy is not None:
            self.policy.to(self.device).eval()
        self._gallery: dict[str, tuple[torch.Tensor, np.ndarray]] = {}
        self._corpus: Corpus | None = None

    @torch.no_grad()
    def prepare(self, corpus: Corpus, batch_size: int = 8) -> None:
        self._corpus = corpus
        names = corpus.gallery_ids()
        self._gallery.clear()
        for start in range(0, len(names), batch_size):
            batch_names = names[start:start + batch_size]
            frames = torch.stack([corpus.video_features(name) for name in batch_names]).to(self.device)
            embeddings, logits = self.model.encode_video(frames, deep=False)
            probabilities = logits.float().softmax(-1).cpu().numpy()
            for i, name in enumerate(batch_names):
                self._gallery[name] = (embeddings[i].detach(), probabilities[i])

    @torch.no_grad()
    def initialize(self, query: Query) -> PoolState:
        if self._corpus is None or not self._gallery:
            raise RuntimeError("Prepare the complete candidate gallery before retrieval.")
        if query.reference_id not in self._gallery:
            raise ValueError("The reference video is absent from the gallery.")
        ref_embedding, ref_probs = self._gallery[query.reference_id]
        text = self._corpus.text_features(query.query_id).to(self.device).unsqueeze(0)
        composed = self.model.compose(ref_embedding.unsqueeze(0), text).squeeze(0)
        candidates = [vid for vid in self._gallery if vid != query.reference_id]
        vectors = torch.stack([self._gallery[vid][0] for vid in candidates])
        similarities = torch.mv(vectors, composed).float().cpu().tolist()
        score_map = dict(zip(candidates, similarities))
        full_initial = tuple(sorted(candidates, key=lambda name: (-score_map[name], name)))
        pool = full_initial[:self.config.pool_size]
        subset_scores = {name: score_map[name] for name in pool}
        state = PoolState(
            pool=pool, initial=pool, full_initial=full_initial,
            scores=subset_scores,
            embeddings={name: self._gallery[name][0] for name in pool},
            probabilities={name: {4: self._gallery[name][1]} for name in pool},
            reference_embedding=ref_embedding,
            reference_probabilities={4: ref_probs}, reviewed=frozenset(),
            bounds=self.ontology.fallback("pending semantic evaluation"),
            evidence={name: State.UNKNOWN for name in pool}, order=list(pool),
        )
        return self._refresh(query, state)

    def _refresh(self, query: Query, state: PoolState) -> PoolState:
        programs = self.grammar.parse(query.modification)
        reference_set = self.calibrator.predict(state.reference_probabilities)
        state.bounds = (self.ontology.fallback("unresolved reference prediction")
                        if reference_set.fallback else self.ontology.bounds(
                            reference_set.labels, programs,
                            self.config.maximum_interpretations, self.config.maximum_conditions))
        candidate_sets = {name: self.calibrator.predict(probs)
                          for name, probs in state.probabilities.items()}
        state.evidence = states(state.bounds, candidate_sets)
        state.order = ranked_order(state.scores, state.evidence, state.initial)
        return state

    @torch.no_grad()
    def review(self, query: Query, state: PoolState, action: Action) -> PoolState:
        if self._corpus is None:
            raise RuntimeError("The gallery has not been prepared.")
        if action.cost <= 0 or action.scope not in {"reference", "candidate"}:
            raise ValueError("Invalid review action.")
        subject = query.reference_id if action.scope == "reference" else action.video_id
        if subject in state.reviewed or (action.scope == "candidate" and subject not in state.pool):
            raise ValueError("Review actions must target an unreviewed pool item or reference.")
        features = self._corpus.video_features(subject).unsqueeze(0).to(self.device)
        embedding, logits = self.model.encode_video(features, deep=True)
        embedding = embedding[0].detach()
        probs12 = logits[0].float().softmax(-1).cpu().numpy()
        updated = replace(
            state, scores=state.scores.copy(), embeddings=state.embeddings.copy(),
            probabilities={name: prediction.copy() for name, prediction in state.probabilities.items()},
            reference_probabilities=state.reference_probabilities.copy(),
            reviewed=state.reviewed | {subject}, evidence=state.evidence.copy(),
            order=state.order.copy(),
        )
        if action.scope == "reference":
            updated.reference_embedding = embedding
            updated.reference_probabilities[12] = probs12
            text = self._corpus.text_features(query.query_id).unsqueeze(0).to(self.device)
            composed = self.model.compose(embedding.unsqueeze(0), text).squeeze(0)
            vectors = torch.stack([updated.embeddings[name] for name in state.pool])
            new_scores = torch.mv(vectors, composed).float().cpu().tolist()
            updated.scores = dict(zip(state.pool, new_scores))
        else:
            updated.embeddings[subject] = embedding
            updated.probabilities[subject][12] = probs12
            text = self._corpus.text_features(query.query_id).unsqueeze(0).to(self.device)
            composed = self.model.compose(updated.reference_embedding.unsqueeze(0), text).squeeze(0)
            updated.scores[subject] = float(torch.dot(embedding, composed).item())
        return self._refresh(query, updated)

    def actions(self, query: Query, state: PoolState) -> tuple[Action, ...]:
        if state.bounds.fallback:
            return ()
        actions = []
        if query.reference_id not in state.reviewed:
            actions.append(Action("reference", query.reference_id, self.reference_cost))
        leading = state.order[:min(len(state.pool), 2 * self.config.output_size)]
        actions.extend(Action("candidate", name, self.candidate_cost)
                       for name in leading if state.evidence[name] == State.UNKNOWN
                       and name not in state.reviewed)
        return tuple(actions)

    def decision_features(self, query: Query, state: PoolState, action: Action,
                          remaining_budget: int) -> torch.Tensor:
        """The twelve policy features in the order specified by the method."""
        n = len(self.ontology.ids)
        is_reference = action.scope == "reference"
        subject = query.reference_id if is_reference else action.video_id
        candidate_set = (self.calibrator.predict(state.reference_probabilities) if is_reference
                         else self.calibrator.predict(state.probabilities[subject]))
        reference_set = self.calibrator.predict(state.reference_probabilities)
        lead = state.order[:min(len(state.pool), 2 * self.config.output_size)]
        unknown_fraction = sum(state.evidence[name] == State.UNKNOWN for name in lead) / max(1, len(lead))
        current_rank = {name: index + 1 for index, name in enumerate(state.order)}
        pairs = eligible_pairs(state.order, state.evidence, self.config.output_size)
        affected = sum(weight for (a, b), weight in pairs.items()
                       if is_reference or subject in (a, b))
        total = sum(pairs.values())
        probs = (state.reference_probabilities if is_reference else state.probabilities[subject])
        p = probs[max(probs)]
        entropy = -sum(float(v * math.log(max(v, 1e-12))) for v in p if v > 0)
        boundary_rank = min(len(state.order), self.config.output_size) - 1
        boundary_score = state.scores[state.order[boundary_rank]] if state.order else 0.0
        if is_reference:
            gap = float(np.mean([abs(state.scores[name] - boundary_score) for name in lead])) if lead else 0.0
        else:
            gap = abs(state.scores[subject] - boundary_score)
        features = [
            float(is_reference), len(reference_set.labels) / n,
            len(candidate_set.labels) / n, len(state.bounds.lower) / n,
            len(state.bounds.upper) / n,
            len(state.bounds.upper - state.bounds.lower) / n,
            entropy / math.log(n), unknown_fraction,
            affected / total if total else 0.0,
            gap / 2, 0.0 if is_reference else current_rank[subject] / self.config.pool_size,
            remaining_budget / max(self.config.review_budget, 1),
        ]
        return torch.tensor(features, dtype=torch.float32)

    @torch.no_grad()
    def infer(self, query: Query) -> tuple[list[str], list[dict], PoolState]:
        if self.policy is None:
            raise RuntimeError("A trained review policy is required for scheduled inference.")
        state = self.initialize(query)
        history: list[dict] = []
        for step in range(self.config.review_budget):
            options = self.actions(query, state)
            if not options:
                break
            features = torch.stack([self.decision_features(query, state, option,
                                                          self.config.review_budget - step)
                                    for option in options]).to(self.device)
            predicted = self.policy(features).cpu().tolist()
            best = max(range(len(options)), key=lambda i: (predicted[i] / options[i].cost, -i))
            if predicted[best] <= 0:
                break
            action = options[best]
            state = self.review(query, state, action)
            history.append({"scope": action.scope, "video_id": action.video_id,
                            "predicted_gain": predicted[best], "cost": action.cost})
        # Ranking changes are confined to the fixed pool; all other gallery items preserve order.
        full_order = state.order + [item for item in state.full_initial if item not in state.pool]
        return full_order, history, state


def eligible_pairs(order: list[str], evidence: Mapping[str, State], k: int) -> dict[tuple[str, str], float]:
    ceiling = min(len(order), 2 * k)
    result = {}
    for i in range(ceiling):
        for j in range(i + 1, ceiling):
            if i >= k and j >= k:
                continue
            left, right = order[i], order[j]
            if State.UNKNOWN not in (evidence[left], evidence[right]):
                continue
            result[(left, right)] = 1 / math.log2(2 + min(i + 1, j + 1))
    return result


def realized_gain(before: PoolState, after: PoolState, positives: frozenset[str],
                  k: int, incorrect_penalty: float = 2.0) -> float:
    """Correct-edge improvement minus weighted incorrect-edge growth (Eq. 9)."""
    pairs = {pair: weight for pair, weight in
             eligible_pairs(before.order, before.evidence, k).items()
             if (pair[0] in positives) != (pair[1] in positives)}
    if not pairs:
        return 0.0
    old, new = edges(before.evidence), edges(after.evidence)
    change = 0.0
    for (a, b), weight in pairs.items():
        expected = (a, b) if a in positives else (b, a)
        reversed_edge = (expected[1], expected[0])
        correct = int(expected in new) - int(expected in old)
        incorrect = int(reversed_edge in new) - int(reversed_edge in old)
        change += weight * (correct - incorrect_penalty * incorrect)
    return change / max(sum(pairs.values()), 1e-8)
