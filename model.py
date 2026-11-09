"""Frozen CLIP input extraction and the dual-budget temporal retrieval model."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from config import Config


class FrozenCLIP(nn.Module):
    """Optional offline feature extraction through a frozen CLIP ViT-B/16 checkpoint."""

    def __init__(self, checkpoint: str = "openai/clip-vit-base-patch16", device: str = "cpu"):
        super().__init__()
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError("Install transformers for raw image and text encoding.") from exc
        self.processor = CLIPProcessor.from_pretrained(checkpoint)
        self.encoder = CLIPModel.from_pretrained(checkpoint).eval().to(device)
        self.device = device
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def images(self, frames: Sequence) -> torch.Tensor:
        inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
        features = self.encoder.get_image_features(pixel_values=inputs["pixel_values"])
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return features.detach().float().cpu()

    @torch.inference_mode()
    def texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
        features = self.encoder.get_text_features(input_ids=inputs["input_ids"],
                                                  attention_mask=inputs["attention_mask"])
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return features.detach().float().cpu()


class OEPModel(nn.Module):
    """Shared 12-slot temporal encoder, class head, and composition/retrieval heads."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        d = config.dimension
        self.positions = nn.Parameter(torch.empty(1, config.slots, d))
        nn.init.trunc_normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimension, dropout=config.dropout,
            batch_first=True, norm_first=False, activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=config.temporal_layers,
                                              enable_nested_tensor=False)
        self.classifier = nn.Linear(d, config.classes)
        self.video_projection = nn.Linear(d, d)
        self.composition = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def encode_video(self, features: torch.Tensor, deep: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1:] != (self.config.slots, self.config.dimension):
            raise ValueError("Video features must have shape [batch, slots, dimension].")
        if not torch.isfinite(features).all():
            raise ValueError("Video features must be finite.")
        active = torch.ones(self.config.slots, dtype=torch.bool, device=features.device)
        if not deep:
            active[:] = False
            active[list(self.config.light_indices)] = True
        mask = (~active).unsqueeze(0).expand(features.shape[0], -1)
        transformed = self.temporal(features + self.positions, src_key_padding_mask=mask)
        pooled = transformed[:, active].mean(dim=1)
        logits = self.classifier(pooled)
        embedding = F.normalize(self.video_projection(pooled), dim=-1)
        return embedding, logits

    def compose(self, reference: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if reference.shape != text.shape or reference.shape[-1] != self.config.dimension:
            raise ValueError("Reference and modification features require the same dimension.")
        text = F.normalize(text, dim=-1)
        return F.normalize(self.composition(torch.cat((reference, text), dim=-1)), dim=-1)

    @staticmethod
    def scores(composed_query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...d,...nd->...n", composed_query, candidates)

    def freeze_classifier_and_temporal(self) -> None:
        """Keep class predictions fixed after split-conformal calibration."""
        for module in (self.temporal, self.classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.positions.requires_grad_(False)

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
