"""Pre-trained model weight registry and download (FR-10)."""

from __future__ import annotations

from pathlib import Path

MODEL_REGISTRY: dict[str, dict[str, str | None]] = {
    "unet_resnet34_mitoem": {
        "description": "U-Net/ResNet34 pre-trained on the MitoEM EM mitochondria dataset",
        "url": None,  # populated once a hosted weights URL is finalized
    },
    "attention_unet_empiar": {
        "description": "Attention U-Net pre-trained on EMPIAR mitochondria datasets",
        "url": None,
    },
}


def list_available_models() -> list[str]:
    """List model names available in the registry."""
    return list(MODEL_REGISTRY.keys())


def download_pretrained_weights(model_name: str, dest_dir: str | Path) -> Path:
    """Download pre-trained weights for ``model_name`` into ``dest_dir``.

    Args:
        model_name: one of :func:`list_available_models`.
        dest_dir: directory to save the downloaded checkpoint into.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list_available_models()}")
    raise NotImplementedError(f"Download URL for '{model_name}' not yet configured; hosted weights pending (FR-10)")
