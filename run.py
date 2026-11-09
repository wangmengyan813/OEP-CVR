"""Training, split calibration, review-policy fitting, evaluation, and reproducibility checks."""
from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import Config, set_seed
from data import Corpus, Query
from model import OEPModel
from reasoning import (ClassSet, ConformalCalibrator, State, average_precision_at_k,
                       holm_adjust, multi_positive_loss, paired_cluster_bootstrap,
                       partial_order_loss, ranked_order, retrieval_metrics, states)
from scheduling import (Action, InferenceEngine, ValueNetwork, realized_gain)
from semantics import Atom, Bounds, Grammar, Ontology, Program, Truth


def build_config(args: argparse.Namespace, classes: int | None = None) -> Config:
    overrides = {item.name: getattr(args, item.name) for item in fields(Config)
                 if getattr(args, item.name, None) is not None}
    if classes is not None:
        overrides["classes"] = classes
    return Config(**overrides)


def candidate_ids(corpus: Corpus, query: Query, maximum_negatives: int,
                  rng: random.Random, evidenced: set[str] | None = None) -> list[str]:
    positive = sorted(vid for vid in query.positives if corpus.videos[vid].split == query.split)
    pool = [vid for vid, record in corpus.videos.items()
            if record.split == query.split and vid not in query.positives and vid != query.reference_id]
    if evidenced is not None:
        pool = [name for name in pool if name in evidenced]
    if not positive or (evidenced is None and not pool):
        raise ValueError(f"Training query lacks split-matched positives or negatives: {query.query_id}")
    negatives = rng.sample(pool, min(maximum_negatives, len(pool)))
    return positive + negatives


def query_loss(model: OEPModel, corpus: Corpus, query: Query,
               names: list[str], stage: str, calibrator: ConformalCalibrator | None,
               ontology: Ontology, grammar: Grammar, config: Config,
               device: torch.device) -> torch.Tensor:
    videos = [query.reference_id] + names
    corpus.require_labels(videos, config.classes)
    frames = torch.stack([corpus.video_features(name) for name in videos]).to(device)
    text = corpus.text_features(query.query_id).unsqueeze(0).to(device)
    embedding4, logits4 = model.encode_video(frames, deep=False)
    candidate_embeddings = embedding4[1:]
    composed = model.compose(embedding4[:1], text)
    scores = (candidate_embeddings * composed).sum(-1)
    gold = torch.tensor([name in query.positives for name in names], device=device)
    retrieval = multi_positive_loss(scores, gold, config.contrastive_temperature)
    if stage == "representation":
        _, logits12 = model.encode_video(frames, deep=True)
        target = torch.tensor([corpus.videos[name].label for name in videos],
                              dtype=torch.long, device=device)
        classification = 0.5 * (F.cross_entropy(logits4, target)
                                + F.cross_entropy(logits12, target))
        return retrieval + config.classification_weight * classification
    if calibrator is None:
        raise RuntimeError("Ranking training requires a fitted conformal calibrator.")
    probabilities = logits4.detach().float().softmax(-1).cpu().numpy()
    reference_set = calibrator.predict({4: probabilities[0]})
    programs = grammar.parse(query.modification)
    bounds = (ontology.fallback("empty calibrated reference") if reference_set.fallback
              else ontology.bounds(reference_set.labels, programs,
                                   config.maximum_interpretations, config.maximum_conditions))
    candidate_sets = {name: calibrator.predict({4: probabilities[i + 1]})
                      for i, name in enumerate(names)}
    semantic_states = states(bounds, candidate_sets)
    ordering = partial_order_loss(scores, names, semantic_states, query.positives,
                                  config.contrastive_temperature)
    return retrieval + config.ordering_weight * ordering


@torch.no_grad()
def evidenced_negative_pools(model: OEPModel, corpus: Corpus, ontology: Ontology,
                            grammar: Grammar, calibrator: ConformalCalibrator,
                            config: Config, device: torch.device) -> dict[str, set[str]]:
    """Retains label-confirmed negatives with conflicting semantic evidence."""
    model.eval()
    class_sets = {}
    names = [name for name, video in corpus.videos.items()
             if video.split in {"train", "validation"}]
    for start in range(0, len(names), 8):
        chunk = names[start:start + 8]
        features = torch.stack([corpus.video_features(name) for name in chunk]).to(device)
        _, logits = model.encode_video(features, deep=False)
        for name, probs in zip(chunk, logits.float().softmax(-1).cpu().numpy()):
            class_sets[name] = calibrator.predict({4: probs})
    pools = {}
    for query in corpus.query_split("train") + corpus.query_split("validation"):
        reference = class_sets.get(query.reference_id)
        bounds = (ontology.fallback("unknown reference") if reference is None or reference.fallback
                  else ontology.bounds(reference.labels, grammar.parse(query.modification),
                                       config.maximum_interpretations, config.maximum_conditions))
        alternatives = {name: prediction for name, prediction in class_sets.items()
                        if corpus.videos[name].split == query.split
                        and name not in query.positives and name != query.reference_id}
        evidence = states(bounds, alternatives)
        pools[query.query_id] = {name for name in alternatives
                                 if evidence[name] == State.CONFLICT}
    return pools


def epoch(model: OEPModel, corpus: Corpus, queries: list[Query], ontology: Ontology,
          grammar: Grammar, config: Config, device: torch.device, stage: str,
          optimizer: torch.optim.Optimizer | None = None,
          calibrator: ConformalCalibrator | None = None, epoch_seed: int = 2022,
          evidenced_pools: dict[str, set[str]] | None = None) -> float:
    if not queries:
        raise ValueError("The training and validation query partitions must not be empty.")
    training = optimizer is not None
    model.train(training)
    if stage == "ranking":
        model.temporal.eval()
    rng = random.Random(epoch_seed)
    ordered = list(queries)
    if training:
        rng.shuffle(ordered)
        optimizer.zero_grad(set_to_none=True)
    total, count = 0.0, 0
    for query in ordered:
        names = candidate_ids(corpus, query, config.maximum_negatives, rng,
                              None if evidenced_pools is None else evidenced_pools[query.query_id])
        with torch.set_grad_enabled(training):
            loss = query_loss(model, corpus, query, names, stage, calibrator,
                              ontology, grammar, config, device)
            if training:
                (loss / config.batch_queries).backward()
        total += float(loss.detach().cpu())
        count += 1
        if training and (count % config.batch_queries == 0 or count == len(ordered)):
            # Correct for a partial terminal accumulation batch.
            remainder = count % config.batch_queries
            if count == len(ordered) and remainder:
                factor = config.batch_queries / remainder
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(factor)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total / count


def early_stop_training(model: OEPModel, corpus: Corpus, ontology: Ontology,
                        grammar: Grammar, config: Config, device: torch.device,
                        stage: str, calibrator: ConformalCalibrator | None,
                        epochs: int) -> dict:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    train, valid = corpus.query_split("train"), corpus.query_split("validation")
    history, best, stale = [], float("inf"), 0
    best_weights = copy.deepcopy(model.state_dict())
    pools = (evidenced_negative_pools(model, corpus, ontology, grammar, calibrator, config, device)
             if stage == "ranking" and calibrator is not None else None)
    for iteration in range(epochs):
        loss_train = epoch(model, corpus, train, ontology, grammar, config, device,
                           stage, optimizer, calibrator, config.seed + iteration, pools)
        loss_valid = epoch(model, corpus, valid, ontology, grammar, config, device,
                           stage, None, calibrator, config.seed + 100_000, pools)
        history.append({"epoch": iteration + 1, "training_loss": loss_train,
                        "validation_loss": loss_valid})
        if loss_valid < best - 1e-7:
            best, stale, best_weights = loss_valid, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_weights)
    return {"stage": stage, "history": history, "best_validation_loss": best}


@torch.no_grad()
def calibrate(model: OEPModel, corpus: Corpus, ontology: Ontology,
              config: Config, device: torch.device) -> ConformalCalibrator:
    names = [video.video_id for video in corpus.videos.values() if video.split == "calibration"]
    if not names:
        raise ValueError("The calibration video partition is empty.")
    corpus.require_labels(names, config.classes)
    model.eval()
    all_probabilities = {4: [], 12: []}
    for start in range(0, len(names), 8):
        chunk = names[start:start + 8]
        frames = torch.stack([corpus.video_features(name) for name in chunk]).to(device)
        for budget in (4, 12):
            _, logits = model.encode_video(frames, deep=(budget == 12))
            all_probabilities[budget].append(logits.float().softmax(-1).cpu().numpy())
    fitted = ConformalCalibrator(ontology.ids, config.alpha)
    fitted.fit({budget: np.concatenate(values) for budget, values in all_probabilities.items()},
               [corpus.videos[name].label for name in names])
    return fitted


@torch.no_grad()
def policy_examples(engine: InferenceEngine, queries: list[Query]) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for query in queries:
        current = engine.initialize(query)
        for step in range(engine.config.review_budget):
            options = engine.actions(query, current)
            if not options:
                break
            scored = []
            for option in options:
                features = engine.decision_features(query, current, option,
                                                     engine.config.review_budget - step)
                updated = engine.review(query, current, option)
                gain = realized_gain(current, updated, query.positives,
                                     engine.config.output_size,
                                     engine.config.incorrect_edge_penalty)
                x.append(features)
                y.append(gain)
                scored.append((gain / option.cost, gain, option, updated))
            winner = max(scored, key=lambda item: item[0])
            if winner[1] <= 0:
                break
            current = winner[3]
    if not x:
        return torch.zeros((0, 12)), torch.zeros((0,))
    return torch.stack(x), torch.tensor(y, dtype=torch.float32)


def train_policy(model: OEPModel, corpus: Corpus, ontology: Ontology,
                 grammar: Grammar, calibrator: ConformalCalibrator,
                 config: Config, device: torch.device) -> tuple[ValueNetwork, dict]:
    engine = InferenceEngine(config, model, ontology, grammar, calibrator, device=str(device))
    engine.prepare(corpus)
    x_train, y_train = policy_examples(engine, corpus.query_split("train"))
    x_valid, y_valid = policy_examples(engine, corpus.query_split("validation"))
    if not len(x_train) or not len(x_valid):
        raise ValueError("No review decisions are available for policy training and validation.")
    policy = ValueNetwork().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    xt, yt = x_train.to(device), y_train.to(device)
    xv, yv = x_valid.to(device), y_valid.to(device)
    best, stale, history = float("inf"), 0, []
    best_weights = copy.deepcopy(policy.state_dict())
    for iteration in range(config.value_epochs):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        objective = F.huber_loss(policy(xt), yt)
        objective.backward()
        optimizer.step()
        policy.eval()
        with torch.no_grad():
            validation_loss = float(F.huber_loss(policy(xv), yv).item())
        history.append({"epoch": iteration + 1, "training_loss": float(objective.item()),
                        "validation_loss": validation_loss})
        if validation_loss < best - 1e-7:
            best, stale, best_weights = validation_loss, 0, copy.deepcopy(policy.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    policy.load_state_dict(best_weights)
    return policy, {"stage": "value", "history": history,
                    "best_validation_loss": best,
                    "training_decisions": len(xt), "validation_decisions": len(xv)}


def train(args: argparse.Namespace) -> None:
    ontology = Ontology.load(args.ontology)
    config = build_config(args, len(ontology.ids))
    set_seed(config.seed)
    device = torch.device(args.device)
    corpus = Corpus(args.data, config.dimension, config.slots)
    model = OEPModel(config).to(device)
    first = early_stop_training(model, corpus, ontology, Grammar({}, []), config,
                                device, "representation", None, config.temporal_epochs)
    model.freeze_classifier_and_temporal()
    calibrator = calibrate(model, corpus, ontology, config, device)
    grammar = Grammar.load(args.grammar, config.maximum_interpretations, config.maximum_conditions)
    second = early_stop_training(model, corpus, ontology, grammar, config, device,
                                 "ranking", calibrator, config.ordering_epochs)
    model.freeze_all()
    value, third = train_policy(model, corpus, ontology, grammar, calibrator, config, device)
    destination = Path(args.checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "model": model.state_dict(),
                "policy": value.state_dict(), "threshold": calibrator.threshold,
                "vocabulary": ontology.ids, "stages": [first, second, third]}, destination)
    print(json.dumps({"checkpoint": str(destination), "calibration_threshold": calibrator.threshold,
                      "training_stages": [first, second, third]}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    ontology = Ontology.load(args.ontology)
    grammar = Grammar.load(args.grammar)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    if tuple(checkpoint["vocabulary"]) != ontology.ids:
        raise ValueError("The checkpoint vocabulary and ontology have different class orders.")
    model = OEPModel(config)
    model.load_state_dict(checkpoint["model"])
    policy = ValueNetwork()
    policy.load_state_dict(checkpoint["policy"])
    calibrator = ConformalCalibrator(ontology.ids, config.alpha)
    calibrator.threshold = float(checkpoint["threshold"])
    corpus = Corpus(args.data, config.dimension, config.slots)
    engine = InferenceEngine(config, model, ontology, grammar, calibrator, policy,
                             device=str(device), reference_cost=args.reference_cost,
                             candidate_cost=args.candidate_cost)
    engine.prepare(corpus)
    results = []
    for query in corpus.query_split(args.split):
        ranking, reviews, final_state = engine.infer(query)
        results.append({"query_id": query.query_id, "source": query.source,
                        "source_video_id": query.source_video_id, "ranking": ranking,
                        "top10": ranking[:10], "positive_count": len(query.positives),
                        "AP@50": average_precision_at_k(ranking, query.positives),
                        "fallback": final_state.bounds.fallback,
                        "fallback_reason": final_state.bounds.reason,
                        "reviews": reviews})
    if not results:
        raise ValueError(f"No queries were found for evaluation split {args.split!r}.")
    metrics = {}
    for source in sorted({entry["source"] for entry in results}):
        subset = [entry for entry in results if entry["source"] == source]
        metrics[source] = retrieval_metrics([entry["ranking"] for entry in subset],
                                            [corpus.queries[entry["query_id"]].positives for entry in subset])
    metrics["all"] = retrieval_metrics([entry["ranking"] for entry in results],
                                       [corpus.queries[entry["query_id"]].positives for entry in results])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"metrics": metrics, "per_query": results}, indent=2),
                           encoding="utf-8")
    print(json.dumps({"output": str(destination), "metrics": metrics}, indent=2))


class MethodChecks(unittest.TestCase):
    """Deterministic mathematical, semantic, and integrated pipeline checks."""

    @classmethod
    def setUpClass(cls):
        cls.ontology = Ontology([
            {"id": "A", "attributes": {"discipline": "gymnastics", "stage": "floor", "twists": 1}},
            {"id": "B", "attributes": {"discipline": "gymnastics", "stage": "floor", "twists": 2}},
            {"id": "C", "attributes": {"discipline": "gymnastics", "twists": 3}},
            {"id": "D", "attributes": {"discipline": "diving", "stage": "platform", "twists": 2}},
        ])
        cls.grammar = Grammar({
            "one more half turn": [[{"op": "add", "attribute": "twists", "value": 1}]],
            "two twists": [[{"op": "set", "attribute": "twists", "value": 2}]],
            "ambiguous stage": [[{"op": "set", "attribute": "stage", "value": "floor"}],
                                [{"op": "set", "attribute": "twists", "value": 2}]],
            "contradiction": [[{"op": "set", "attribute": "twists", "value": 2},
                               {"op": "negate", "attribute": "twists", "value": 2}]],
        })

    def test_configuration_and_geometry(self):
        config = Config()
        self.assertEqual(config.light_slots, 4)
        self.assertEqual(config.slots, 12)
        self.assertEqual(len(set(config.light_indices)), 4)
        with self.assertRaises(ValueError):
            Config(dimension=30, attention_heads=8)

    def test_ontology_and_unknown_attributes(self):
        self.assertEqual(self.ontology.evaluate("B", "A", Atom("add", "twists", 1)), Truth.TRUE)
        self.assertEqual(self.ontology.evaluate("C", "A", Atom("set", "stage", "floor")), Truth.UNKNOWN)
        self.assertEqual(self.ontology.evaluate("D", "A", Atom("set", "stage", "floor")), Truth.FALSE)

    def test_bounds_and_missing_semantics(self):
        lower = self.ontology.bounds(frozenset({"A"}), self.grammar.parse("one more half turn"))
        self.assertEqual(lower.lower, frozenset({"B", "D"}))
        self.assertEqual(lower.upper, frozenset({"B", "D"}))
        uncertainty = self.ontology.bounds(frozenset({"A"}), self.grammar.parse("ambiguous stage"))
        self.assertEqual(uncertainty.lower, frozenset({"B"}))
        self.assertTrue({"A", "B", "C"}.issubset(uncertainty.upper))

    def test_invalid_program_and_fallback(self):
        self.assertFalse(self.grammar.parse("contradiction"))
        self.assertFalse(self.grammar.parse("not in grammar"))
        self.assertTrue(self.ontology.bounds(frozenset({"A"}), ()).fallback)
        self.assertTrue(self.ontology.bounds(frozenset({"X"}), self.grammar.parse("two twists")).fallback)

    def test_conformal_joint_calibration(self):
        n = 20
        light = np.repeat([[0.80, 0.10, 0.05, 0.05]], n, axis=0)
        deep = np.repeat([[0.65, 0.15, 0.10, 0.10]], n, axis=0)
        calibration = ConformalCalibrator(self.ontology.ids, alpha=0.10)
        self.assertAlmostEqual(calibration.fit({4: light, 12: deep}, [0] * n), 0.35)
        self.assertEqual(calibration.predict({4: light[0], 12: deep[0]}).labels, frozenset({"A"}))
        self.assertTrue(calibration.predict({4: np.array([0.25] * 4),
                                             12: np.array([1.0, 0, 0, 0])}).fallback)

    def test_finite_sample_infinity(self):
        calibration = ConformalCalibrator(self.ontology.ids, 0.05)
        probabilities = np.repeat([[0.7, 0.1, 0.1, 0.1]], 3, axis=0)
        self.assertEqual(calibration.fit({4: probabilities, 12: probabilities}, [0] * 3), float("inf"))
        self.assertEqual(len(calibration.predict({4: probabilities[0]}).labels), 4)

    def test_noncompensatory_precedence(self):
        scores = {"support": 0.1, "unknown": 0.8, "conflict": 0.9}
        labels = {"support": State.SUPPORT, "unknown": State.UNKNOWN, "conflict": State.CONFLICT}
        ordered = ranked_order(scores, labels, list(scores))
        self.assertLess(ordered.index("support"), ordered.index("conflict"))
        self.assertEqual(ranked_order(scores, {name: State.UNKNOWN for name in scores},
                                      list(scores)), list(scores))

    def test_partial_order_training_gradient(self):
        scores = torch.tensor([0.2, 0.7], requires_grad=True)
        evidence = {"P": State.SUPPORT, "N": State.CONFLICT}
        loss = partial_order_loss(scores, ["P", "N"], evidence, frozenset({"P"}))
        loss.backward()
        self.assertLess(float(scores.grad[0]), 0)
        self.assertGreater(float(scores.grad[1]), 0)
        self.assertGreater(multi_positive_loss(scores.detach(), torch.tensor([True, False])), 0)

    def test_retrieval_metrics(self):
        positives = frozenset({"X", "Y"})
        self.assertAlmostEqual(average_precision_at_k(["X", "N", "Y"], positives, 50),
                               (1 + 2 / 3) / 2)
        metrics = retrieval_metrics([["X", "N", "Y"]], [positives])
        self.assertEqual(metrics["Hit@10"], 100.0)
        self.assertEqual(metrics["Recall@10"], 100.0)

    def test_cluster_bootstrap_and_holm(self):
        outcome = paired_cluster_bootstrap(np.array([0.4, 0.5, 0.8]),
                                          np.array([0.3, 0.4, 0.7]),
                                          ["a", "a", "b"], 80, seed=4)
        self.assertAlmostEqual(outcome["delta_pp"], 10)
        self.assertEqual(holm_adjust([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06])

    def test_model_dimensions_and_backpropagation(self):
        set_seed(37)
        config = Config(dimension=16, classes=4, attention_heads=4,
                        feedforward_dimension=32, dropout=0.0)
        model = OEPModel(config)
        frames = torch.randn((2, 12, 16))
        encoded, logits = model.encode_video(frames)
        self.assertEqual(encoded.shape, (2, 16))
        self.assertEqual(logits.shape, (2, 4))
        query = model.compose(encoded[:1], torch.randn((1, 16)))
        loss = model.scores(query, encoded[1:].unsqueeze(0)).sum() + logits.sum()
        loss.backward()
        self.assertTrue(model.positions.grad is not None)

    @staticmethod
    def _small_corpus(root: Path) -> Corpus:
        videos = [
            ("r", 0, (1., 0.), (0., 1.)),
            ("p", 1, (1., 0.), (0.8, 0.2)),
            ("n", 2, (0., 1.), (1., 0.)),
            ("d", 3, (0.8, 0.2), (0.1, 0.9)),
            ("z", 0, (0.2, 0.8), (0.8, 0.2)),
        ]
        with (root / "videos.jsonl").open("w", encoding="utf-8") as stream:
            for name, label, light, deep in videos:
                array = np.zeros((12, 16), dtype=np.float32)
                array[0, 0] = label
                array[0, 1:3] = light
                array[11, 1:3] = deep
                np.save(root / f"{name}.npy", array)
                stream.write(json.dumps({"video_id": name, "feature_file": f"{name}.npy",
                                         "label": label, "split": "test", "source_video_id": name,
                                         "competition_id": name}) + "\n")
        np.save(root / "text.npy", np.array([0., 1.] + [0.] * 14, dtype=np.float32))
        with (root / "queries.jsonl").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"query_id": "q", "reference_id": "r",
                                     "modification": "two twists", "text_file": "text.npy",
                                     "positives": ["p", "d"], "split": "test", "source": "FineGym",
                                     "source_video_id": "r", "competition_id": "meet"}) + "\n")
        return Corpus(root, 16, 12)

    @staticmethod
    def _integration_engine(corpus: Corpus, policy: torch.nn.Module | None = None):
        class FixedEncoder(torch.nn.Module):
            def encode_video(self, frames, deep=False):
                class_id = frames[:, 0, 0].long()
                active = frames[:, 11 if deep else 0, 1:3]
                features = F.pad(F.normalize(active, dim=-1), (0, 14))
                logits = torch.full((len(frames), 4), -8.0, device=frames.device)
                logits.scatter_(1, class_id[:, None], 8.0)
                return features, logits

            def compose(self, reference, text):
                return F.normalize(reference + text, dim=-1)

        config = Config(classes=4, dimension=16, attention_heads=4,
                        feedforward_dimension=32, pool_size=4, output_size=2,
                        review_budget=2)
        calibration = ConformalCalibrator(("A", "B", "C", "D"), 0.10)
        calibration.threshold = 0.20
        engine = InferenceEngine(config, FixedEncoder(), MethodChecks.ontology,
                                 MethodChecks.grammar, calibration, policy)
        engine.prepare(corpus)
        return engine

    def test_reference_review_refreshes_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = self._small_corpus(Path(folder))
            engine = self._integration_engine(corpus)
            query = corpus.queries["q"]
            before = engine.initialize(query)
            self.assertEqual(before.evidence["p"], State.SUPPORT)
            self.assertEqual(before.evidence["n"], State.CONFLICT)
            self.assertLess(before.order.index("p"), before.order.index("n"))
            action = Action("reference", "r", 1.0)
            after = engine.review(query, before, action)
            changed = [name for name in before.pool
                       if not np.isclose(before.scores[name], after.scores[name])]
            self.assertGreaterEqual(len(changed), 2)
            self.assertEqual(set(before.pool), set(after.pool))
            self.assertIn("r", after.reviewed)
            self.assertNotIn("r", before.reviewed)

    def test_candidate_review_changes_only_one_score(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = self._small_corpus(Path(folder))
            engine = self._integration_engine(corpus)
            query = corpus.queries["q"]
            before = engine.initialize(query)
            after = engine.review(query, before, Action("candidate", "p", 1.0))
            self.assertNotAlmostEqual(before.scores["p"], after.scores["p"])
            for name in before.pool:
                if name != "p":
                    self.assertEqual(before.scores[name], after.scores[name])
            with self.assertRaises(ValueError):
                engine.review(query, after, Action("candidate", "p", 1.0))

    def test_unparsed_query_retains_initial_order(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as folder:
            corpus = self._small_corpus(Path(folder))
            class ConstantPolicy(torch.nn.Module):
                def forward(self, values):
                    return torch.ones(values.shape[0], device=values.device)
            engine = self._integration_engine(corpus, ConstantPolicy())
            query = replace(corpus.queries["q"], modification="unlisted instruction")
            ranking, history, state = engine.infer(query)
            self.assertTrue(state.bounds.fallback)
            self.assertEqual(history, [])
            self.assertEqual(ranking, list(state.full_initial))

    def test_review_limit_and_policy_features(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = self._small_corpus(Path(folder))
            class ConstantPolicy(torch.nn.Module):
                def forward(self, values):
                    return torch.ones(values.shape[0], device=values.device)
            engine = self._integration_engine(corpus, ConstantPolicy())
            query = corpus.queries["q"]
            state = engine.initialize(query)
            options = engine.actions(query, state)
            self.assertGreater(len(options), 0)
            for action in options:
                features = engine.decision_features(query, state, action, 2)
                self.assertEqual(features.shape, (12,))
                self.assertTrue(torch.isfinite(features).all())
            ranking, history, final = engine.infer(query)
            self.assertLessEqual(len(history), 2)
            self.assertEqual(len(set(ranking)), len(corpus.videos) - 1)
            self.assertEqual(set(ranking), set(corpus.videos) - {"r"})
            self.assertIn("r", final.reviewed)

    def test_evidenced_negative_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = self._small_corpus(Path(folder))
            query = corpus.queries["q"]
            selected = candidate_ids(corpus, query, 5, random.Random(42), {"n"})
            self.assertEqual(set(selected), {"p", "d", "n"})
            self.assertEqual(candidate_ids(corpus, query, 5, random.Random(42), set()),
                             ["d", "p"])

    def test_relational_gain_uses_labeled_pairs(self):
        from types import SimpleNamespace
        before = SimpleNamespace(order=["p", "n", "x"],
                                 evidence={"p": State.UNKNOWN, "n": State.UNKNOWN,
                                           "x": State.UNKNOWN})
        after = SimpleNamespace(order=["p", "n", "x"],
                                evidence={"p": State.SUPPORT, "n": State.CONFLICT,
                                          "x": State.UNKNOWN})
        self.assertAlmostEqual(realized_gain(before, after, frozenset({"p"}), 2), 0.5)
        self.assertAlmostEqual(realized_gain(after, after, frozenset({"p"}), 2), 0.0)

    def test_grammar_regex_and_subclass_entailment(self):
        parent = Ontology([
            {"id": "P", "attributes": {"discipline": "gymnastics"}},
            {"id": "C", "parents": ["P"], "attributes": {"twists": 2}},
        ])
        self.assertEqual(parent.value("C", "discipline"), "gymnastics")
        grammar = Grammar({}, [{"pattern": r"show (?P<number>\d+) twists",
                               "programs": [[{"op": "set", "attribute": "twists",
                                              "value": "{number}"}]]}])
        programs = grammar.parse("show 2 twists")
        self.assertEqual(programs[0].atoms[0].value, 2)
        result = parent.bounds(frozenset({"P"}), programs)
        self.assertEqual(result.lower, frozenset({"C"}))

    def test_frozen_classifier_remains_calibrated(self):
        set_seed(73)
        config = Config(classes=4, dimension=16, attention_heads=4,
                        feedforward_dimension=32, dropout=0.0)
        model = OEPModel(config)
        model.freeze_classifier_and_temporal()
        batch = torch.randn((2, 12, 16))
        model.temporal.eval()
        _, before = model.encode_video(batch)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
        optimizer.zero_grad()
        embeddings, _ = model.encode_video(batch)
        loss = embeddings.square().mean() + model.compose(
            embeddings[:1], torch.randn(1, 16)).sum()
        loss.backward()
        optimizer.step()
        _, after = model.encode_video(batch)
        self.assertTrue(torch.equal(before, after))

    def test_source_group_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (root / "videos.jsonl").open("w") as stream:
                for vid in ("r", "p", "n"):
                    stream.write(json.dumps({"video_id": vid, "feature_file": "f.npy",
                                             "label": 0, "split": "train", "source_video_id": "sv",
                                             "competition_id": "cp"}) + "\n")
            with (root / "queries.jsonl").open("w") as stream:
                for index, split in enumerate(("train", "validation")):
                    stream.write(json.dumps({"query_id": str(index), "reference_id": "r",
                                             "modification": "two twists", "text_file": "t.npy",
                                             "positives": ["p"], "split": split,
                                             "source": "FineGym", "source_video_id": "sv",
                                             "competition_id": "cp"}) + "\n")
            with self.assertRaisesRegex(ValueError, "cross data partitions"):
                Corpus(root, 16)


def self_test() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MethodChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="OEP-CVR reproducible research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test", help="Run deterministic mathematical and pipeline checks")
    for command in ("validate-data", "train", "evaluate"):
        cmd = commands.add_parser(command)
        cmd.add_argument("--data", required=True)
        cmd.add_argument("--ontology", required=True)
        cmd.add_argument("--grammar", required=True)
        if command != "validate-data":
            cmd.add_argument("--device", default="cpu")
            cmd.add_argument("--checkpoint", required=True)
        if command == "evaluate":
            cmd.add_argument("--split", default="test")
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--reference-cost", type=float, default=1.0)
            cmd.add_argument("--candidate-cost", type=float, default=1.0)
        if command == "train":
            for flag, convert in (("seed", int), ("dimension", int),
                                  ("temporal_epochs", int), ("ordering_epochs", int),
                                  ("value_epochs", int), ("batch_queries", int),
                                  ("patience", int)):
                cmd.add_argument("--" + flag.replace("_", "-"), type=convert)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "validate-data":
        ontology = Ontology.load(args.ontology)
        grammar = Grammar.load(args.grammar)
        corpus = Corpus(args.data)
        for item in corpus.videos.values():
            corpus.video_features(item.video_id)
        for item in corpus.queries.values():
            corpus.text_features(item.query_id)
        print(json.dumps({"videos": len(corpus.videos), "queries": len(corpus.queries),
                          "ontology_classes": len(ontology.ids),
                          "queries_with_valid_interpretations": sum(
                              bool(grammar.parse(q.modification)) for q in corpus.queries.values())}, indent=2))
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
