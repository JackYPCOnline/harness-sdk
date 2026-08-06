"""Model routing primitives.

A ``RoutingStrategy`` returns the candidates to use once per invocation, most preferred first, and
``ModelRouter`` executes that sequence: it uses the first entry and advances through the rest when a
model fails and no hook has claimed the retry. The strategy therefore owns fallback scope -- the
default ``FallbackStrategy`` returns every candidate in declaration order, while a strategy that
returns one candidate disables failover. The API is provisional and may change before it is finalized.
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
