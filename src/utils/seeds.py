# src/utils/seeds.py
import zlib
import numpy as np


def derive_rng(*parts) -> np.random.Generator:
    """
    Independent Generator from a tuple of ints/strings.

    Distinct `parts` -> statistically independent streams, so
    derive_rng(1000, 5, "dates") and derive_rng(1000, 5, "context")
    share nothing. Replaces additive seed offsets, which alias.
    """
    ints = [
        p if isinstance(p, (int, np.integer))
        else zlib.crc32(str(p).encode())
        for p in parts
    ]
    return np.random.default_rng(np.random.SeedSequence(ints))