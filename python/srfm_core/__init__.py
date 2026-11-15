"""srfm-core: the SRFM bar transforms from srfm-lab.

Beta (Minkowski TIMELIKE/SPACELIKE), black-hole mass, the rolling geodesic
and the Hawking z proxy, as streaming classes and as numpy/pandas batch
functions. The same math ships as the Rust crate ``srfm-core``; both are
checked against one set of golden vectors generated from srfm-lab's
``lib/srfm_core.py``.

These are feature transforms. They are not a trading edge on their own.
"""

from .frames import add_srfm_columns, beta, bh_mass, geodesic, hawking_z, wilder_atr
from .kernels import BH_DEFAULTS
from .streaming import (
    SPACELIKE,
    TIMELIKE,
    UNKNOWN,
    BlackHoleDetector,
    GeodesicAnalyzer,
    GravitationalLens,
    HawkingMonitor,
    MinkowskiClassifier,
    WilderATR,
)

__version__ = "0.1.1"

__all__ = [
    "beta",
    "bh_mass",
    "geodesic",
    "hawking_z",
    "wilder_atr",
    "add_srfm_columns",
    "BH_DEFAULTS",
    "MinkowskiClassifier",
    "BlackHoleDetector",
    "GeodesicAnalyzer",
    "GravitationalLens",
    "HawkingMonitor",
    "WilderATR",
    "TIMELIKE",
    "SPACELIKE",
    "UNKNOWN",
    "__version__",
]
