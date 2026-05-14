"""
Experimental modules — not production-ready.

Importing from this package emits a FutureWarning so integration tests and
static analysis pipelines surface any accidental production dependency on
these modules.
"""

import warnings

warnings.warn(
    "quotadrift._experimental contains modules that are not production-hardened. "
    "Do not import from this package in production code paths.",
    FutureWarning,
    stacklevel=2,
)
