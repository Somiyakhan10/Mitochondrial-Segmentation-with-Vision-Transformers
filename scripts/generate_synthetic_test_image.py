"""Generate a synthetic multi-channel TIFF for testing the dashboard/CLI.

Not part of the pipeline itself - a throwaway helper since no real lab
microscopy data exists yet. Produces a 2-channel image with round,
mitochondria-like blobs on a "Tom20" channel and diffuse blob regions on
a "NeuN" channel, with real OME channel-name metadata so
mitomorph.preprocessing.channel_utils can identify them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import disk


def generate(size: int = 512, n_blobs: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    mito_channel = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_blobs):
        cy, cx = rng.integers(20, size - 20, size=2)
        radius = rng.integers(3, 10)
        rr, cc = disk((cy, cx), radius, shape=mito_channel.shape)
        mito_channel[rr, cc] = rng.uniform(0.6, 1.0)
    mito_channel += rng.normal(0, 0.03, mito_channel.shape).clip(0, None)

    neuron_channel = np.zeros((size, size), dtype=np.float32)
    rr, cc = disk((size // 2, size // 2), size // 3, shape=neuron_channel.shape)
    neuron_channel[rr, cc] = 0.5
    neuron_channel += rng.normal(0, 0.05, neuron_channel.shape).clip(0, None)

    stack = np.stack([mito_channel, neuron_channel])
    return (np.clip(stack, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/synthetic_test_image.ome.tif")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n-blobs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = generate(size=args.size, n_blobs=args.n_blobs, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(output_path),
        data,
        metadata={"axes": "CYX", "Channel": {"Name": ["Tom20", "NeuN"]}},
    )
    print(f"Wrote synthetic test image to {output_path} (shape={data.shape})")


if __name__ == "__main__":
    main()
