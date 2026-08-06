"""Convenience script: generate a comprehensive PDF report.

Thin wrapper around ``mitomorph.cli report``.
"""

from __future__ import annotations

import sys

from mitomorph.cli import cli

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "report", *sys.argv[1:]]
    cli()
