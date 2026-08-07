"""Shared pytest configuration.

Force matplotlib's non-interactive Agg backend before any test imports
pyplot. Without this, matplotlib defaults to an interactive backend
(e.g. TkAgg) when Tk is available, which is unreliable when many
figures are created across many tests in a single process and causes
intermittent Tk-related failures.
"""

import matplotlib

matplotlib.use("Agg")
