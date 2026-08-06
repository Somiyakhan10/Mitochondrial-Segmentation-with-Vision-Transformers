"""Sanity checks on extracted morphometric measurements (NFR-07).

NFR-07 requires measurements be reproducible with coefficient of
variation < 5%; that's validated empirically once real data exists. This
module instead flags individual measurements that fall outside physically
plausible ranges, which is checkable without any real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mitomorph.morphometrics.single_features import SingleMitoFeatures


@dataclass
class FeatureQCResult:
    label: int
    is_valid: bool
    flags: list[str] = field(default_factory=list)


def validate_features(
    features: list[SingleMitoFeatures],
    min_area: float = 1.0,
    max_aspect_ratio: float = 50.0,
    max_circularity: float = 1.05,
) -> list[FeatureQCResult]:
    """Flag morphometric measurements outside plausible physical ranges."""
    results = []
    for f in features:
        flags = []
        if f.area < min_area:
            flags.append(f"area {f.area:.3f} below minimum {min_area}")
        if f.aspect_ratio > max_aspect_ratio:
            flags.append(f"aspect_ratio {f.aspect_ratio:.2f} exceeds maximum {max_aspect_ratio}")
        if f.circularity > max_circularity:
            flags.append(f"circularity {f.circularity:.3f} exceeds physical maximum {max_circularity}")
        if not (0.0 <= f.solidity <= 1.0):
            flags.append(f"solidity {f.solidity:.3f} out of [0, 1] range")
        results.append(FeatureQCResult(label=f.label, is_valid=not flags, flags=flags))
    return results
