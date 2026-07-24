"""Model routing: choose which model handles a call at runtime.

This package is internal while the API is stabilized; public exports are added later.
"""

from .router import CandidateInput, ModelRouter, RoutingCandidate
from .strategy import RoutingContext, RoutingStrategy

__all__ = [
    "CandidateInput",
    "ModelRouter",
    "RoutingCandidate",
    "RoutingContext",
    "RoutingStrategy",
]
