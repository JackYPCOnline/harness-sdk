"""Model routing primitives.

``ModelRouter`` holds an ordered set of candidate models and, per invocation, selects one via a
``RoutingStrategy`` (default: ``FallbackStrategy``). When the selected model's retries are
exhausted, the router advances to the next candidate in declaration order. The API is provisional
and may change before it is finalized.
"""

from .router import CandidateInput, FallbackStrategy, ModelRouter, RoutingCandidate
from .strategy import RoutingContext, RoutingStrategy

__all__ = [
    "CandidateInput",
    "FallbackStrategy",
    "ModelRouter",
    "RoutingCandidate",
    "RoutingContext",
    "RoutingStrategy",
]
