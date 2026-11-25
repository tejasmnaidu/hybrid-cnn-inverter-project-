"""Load project CNN checkpoints without requiring the training dataset."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import torch

from preprocessing.preprocess import NormalizationStats

from .model_factory import create_model


def load_checkpoint_for_inference(
    checkpoint_source: str | Path | BinaryIO,
    device: torch.device | str,
) -> tuple[torch.nn.Module, NormalizationStats, dict, dict]:
    """Load architecture, trained weights, and embedded normalization metadata."""
    checkpoint = torch.load(
        checkpoint_source,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("The selected file is not a project training checkpoint.")

    metadata = checkpoint.get("metadata", {})
    model_config = metadata.get("model_config")
    normalization_data = metadata.get("normalization")
    if not isinstance(model_config, dict) or not isinstance(normalization_data, dict):
        raise ValueError("Checkpoint model or normalization metadata is missing.")

    normalization = NormalizationStats(**normalization_data)
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, normalization, metadata, checkpoint
