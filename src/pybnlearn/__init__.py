"""pybnlearn: a Python port of the R package bnlearn.

The numerical core is bnlearn's own C code, compiled against a compatibility
layer that stands in for the R C API, so results agree with R rather than
merely resembling them.  See NOTICE for attribution and licensing.
"""

from ._core import BNLearnError, ci_test

__all__ = ["BNLearnError", "ci_test"]

__version__ = "0.1.0.dev0"
