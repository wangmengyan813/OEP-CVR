"""Strict TF-CoVR manifest contracts and CLIP-feature storage."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class Video:
    video_id: str
    feature_file: str
    label: int | None
    split: str
    source_video_id: str
    competition_id: str
    source: str


@dataclass(frozen=True)
class Query:
    query_id: str
    reference_id: str
    modification: str
    text_file: str
    positives: frozenset[str]
    split: str
    source: str
    source_video_id: str
    competition_id: str


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            yield record


class Corpus:
    """Loads mutually consistent query, video, and relevance manifests."""

    def __init__(self, root: str | Path, dimension: int = 512, slots: int = 12):
        self.root = Path(root).expanduser().resolve()
        self.dimension, self.slots = dimension, slots
        self.videos = {}
        self.queries = {}
        for item in _read_jsonl(self.root / "videos.jsonl"):
            video = Video(
                video_id=str(item["video_id"]), feature_file=str(item["feature_file"]),
                label=None if item.get("label") is None else int(item["label"]),
                split=str(item.get("split", "gallery")),
                source_video_id=str(item["source_video_id"]),
                competition_id=str(item["competition_id"]),
                source=str(item.get("source", "shared")),
            )
            if video.video_id in self.videos:
                raise ValueError(f"Repeated video ID: {video.video_id}")
            self.videos[video.video_id] = video
        for item in _read_jsonl(self.root / "queries.jsonl"):
            query = Query(
                query_id=str(item["query_id"]), reference_id=str(item["reference_id"]),
                modification=str(item["modification"]), text_file=str(item["text_file"]),
                positives=frozenset(map(str, item["positives"])),
                split=str(item["split"]), source=str(item["source"]),
                source_video_id=str(item["source_video_id"]),
                competition_id=str(item["competition_id"]),
            )
            if query.query_id in self.queries:
                raise ValueError(f"Repeated query ID: {query.query_id}")
            if query.reference_id not in self.videos:
                raise ValueError(f"Missing query reference: {query.query_id}")
            if (not query.positives or not query.positives.issubset(self.videos.keys())
                    or query.reference_id in query.positives):
                raise ValueError(f"Missing or invalid positives: {query.query_id}")
            self.queries[query.query_id] = query
        if not self.queries or not self.videos:
            raise ValueError("The manifests must contain videos and queries.")
        self._validate_group_partitions()

    def _validate_group_partitions(self) -> None:
        source_assignments, competition_assignments = {}, {}
        for video in self.videos.values():
            if video.split not in {"train", "validation", "calibration"}:
                continue
            for table, key in ((source_assignments, (video.source, video.source_video_id)),
                               (competition_assignments, (video.source, video.competition_id))):
                old = table.setdefault(key, video.split)
                if old != video.split:
                    raise ValueError("Source-video or competition groups cross data partitions.")
        # Query-level groups are checked independently of gallery video partitions.
        source_assignments, competition_assignments = {}, {}
        for q in self.queries.values():
            if q.split not in {"train", "validation", "calibration"}:
                continue
            source_key = (q.source, q.source_video_id)
            competition_key = (q.source, q.competition_id)
            for table, key in ((source_assignments, source_key),
                               (competition_assignments, competition_key)):
                old = table.setdefault(key, q.split)
                if old != q.split:
                    raise ValueError("Source-video or competition groups cross data partitions.")

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Feature paths must remain inside the dataset directory.")
        return path

    @lru_cache(maxsize=8192)
    def video_features(self, video_id: str) -> torch.Tensor:
        array = np.load(self._path(self.videos[video_id].feature_file), allow_pickle=False)
        if array.shape != (self.slots, self.dimension) or not np.isfinite(array).all():
            raise ValueError(f"Invalid frame features: {video_id}; expected [{self.slots}, {self.dimension}].")
        return torch.tensor(array, dtype=torch.float32)

    @lru_cache(maxsize=8192)
    def text_features(self, query_id: str) -> torch.Tensor:
        array = np.load(self._path(self.queries[query_id].text_file), allow_pickle=False)
        if array.shape != (self.dimension,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid text features: {query_id}; expected [{self.dimension}].")
        return torch.tensor(array, dtype=torch.float32)

    def query_split(self, split: str) -> list[Query]:
        return [q for q in self.queries.values() if q.split == split]

    def gallery_ids(self) -> list[str]:
        return sorted(self.videos)

    def require_labels(self, video_ids: list[str], classes: int) -> None:
        for video_id in video_ids:
            label = self.videos[video_id].label
            if label is None or not 0 <= label < classes:
                raise ValueError(f"A valid training/calibration class label is required: {video_id}")
