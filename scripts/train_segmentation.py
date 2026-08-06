"""Convenience script: fine-tune the segmentation model.

Thin wrapper around ``mitomorph.cli train``.
"""

from __future__ import annotations

import sys

from mitomorph.cli import cli

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "train", *sys.argv[1:]]
    cli()
