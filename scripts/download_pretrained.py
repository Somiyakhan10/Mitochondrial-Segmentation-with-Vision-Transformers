"""Convenience script: download pre-trained segmentation model weights (FR-10)."""

from __future__ import annotations

import argparse

from mitomorph.segmentation.model_zoo import download_pretrained_weights, list_available_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", choices=list_available_models())
    parser.add_argument("--dest-dir", default="data/models")
    args = parser.parse_args()
    download_pretrained_weights(args.model_name, args.dest_dir)


if __name__ == "__main__":
    main()
