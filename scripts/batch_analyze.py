"""Convenience script: batch-analyze a directory of images.

Thin wrapper around ``mitomorph.cli batch`` for running
``python scripts/batch_analyze.py <input_dir> --output-dir <output_dir>``
without installing the package's console script.
"""

from __future__ import annotations

import sys

from mitomorph.cli import cli

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "batch", *sys.argv[1:]]
    cli()
