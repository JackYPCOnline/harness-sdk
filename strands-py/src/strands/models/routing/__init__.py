"""Model routing primitives.

A ``RoutingStrategy`` selects candidates in preference order once per invocation, and ``ModelRouter``
routes across them. The default ``FallbackStrategy`` selects in declaration order. When a model fails
and no hook has claimed the retry, the router advances to the next candidate. The API is provisional
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
