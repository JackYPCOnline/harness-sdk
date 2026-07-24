"""ModelRouter: a reusable, immutable routing profile over candidate models.

A router holds an ordered set of candidates and a strategy. It is a ``Plugin`` so an agent
can accept it through ``model=`` and later phases can install selection middleware. In this
phase the router normalizes candidates, exposes the first candidate resolved to a concrete
model for ``agent.model``, and rejects stateful candidates; it does not yet select per call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union

from ...plugins.plugin import Plugin
from ..bedrock import BedrockModel
from ..model import Model
from .strategy import RoutingStrategy


@dataclass(frozen=True)
class RoutingCandidate:
    """A routing candidate: a model plus optional selection metadata.

    ``name`` and ``description`` are optional. Order- and metric-based strategies ignore
    them; a semantic strategy (e.g. a judge model) requires unique names and descriptions
    to form a classification contract.
    """

    model: Model | str | ModelRouter
    name: str | None = None
    description: str | None = None


CandidateInput = Union[Model, str, "ModelRouter", RoutingCandidate]

_ROUTER_PLUGIN_NAME = "strands:model-router"


class ModelRouter(Plugin):
    """A reusable routing profile: ordered candidate models plus a selection strategy."""

    def __init__(self, models: Sequence[CandidateInput], strategy: RoutingStrategy) -> None:
        """Initialize the router.

        Args:
            models: Candidates as a sequence. Each is a ``Model``, a model-id string
                (resolved to a ``BedrockModel``), a nested ``ModelRouter``, or a
                ``RoutingCandidate`` carrying an optional name/description. The first
                candidate is the router's default.
            strategy: The strategy that chooses a candidate per call.

        Raises:
            TypeError: If ``models`` is not a sequence of candidates.
            ValueError: If ``models`` is empty, candidate names collide, or any candidate is
                a stateful model.
        """
        super().__init__()

        candidates = self._normalize(models)
        if not candidates:
            raise ValueError("ModelRouter requires at least one candidate model")
        self._reject_stateful(candidates)
        self._reject_duplicate_names(candidates)

        self._candidates = candidates
        self._strategy = strategy

    @property
    def name(self) -> str:
        """Stable plugin identifier."""
        return _ROUTER_PLUGIN_NAME

    @property
    def strategy(self) -> RoutingStrategy:
        """The configured selection strategy."""
        return self._strategy

    @property
    def candidates(self) -> tuple[RoutingCandidate, ...]:
        """The router's normalized candidates, in declaration order."""
        return self._candidates

    @property
    def default_model(self) -> Model:
        """The first candidate resolved to a concrete model (recursing nested routers)."""
        return self._resolve(self._candidates[0].model)

    @classmethod
    def _resolve(cls, model: Model | str | ModelRouter) -> Model:
        if isinstance(model, ModelRouter):
            return model.default_model
        if isinstance(model, str):  # normalization resolves strings; defensive fallback
            return BedrockModel(model_id=model)
        return model

    @classmethod
    def _normalize(cls, models: object) -> tuple[RoutingCandidate, ...]:
        # A mapping is a Sequence-like but name-keyed shape we intentionally do not accept;
        # names live on RoutingCandidate. str/bytes are Sequences too but not candidate lists.
        if isinstance(models, (str, bytes, Mapping)) or not isinstance(models, Sequence):
            raise TypeError("models must be a sequence of candidates")
        return tuple(cls._as_candidate(item) for item in models)

    @classmethod
    def _as_candidate(cls, item: CandidateInput) -> RoutingCandidate:
        if isinstance(item, RoutingCandidate):
            return RoutingCandidate(
                model=cls._resolve_shorthand(item.model), name=item.name, description=item.description
            )
        return RoutingCandidate(model=cls._resolve_shorthand(item))

    @staticmethod
    def _resolve_shorthand(model: Model | str | ModelRouter) -> Model | ModelRouter:
        if isinstance(model, str):
            return BedrockModel(model_id=model)
        return model

    @classmethod
    def _reject_stateful(cls, candidates: tuple[RoutingCandidate, ...]) -> None:
        # Nested routers already validated their own candidates at construction.
        for candidate in candidates:
            if isinstance(candidate.model, Model) and candidate.model.stateful:
                label = candidate.name or cls._model_id(candidate.model) or "candidate"
                raise ValueError(f"candidate=<{label}> is stateful; routing among stateful models is not supported")

    @staticmethod
    def _reject_duplicate_names(candidates: tuple[RoutingCandidate, ...]) -> None:
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.name is None:
                continue
            if candidate.name in seen:
                raise ValueError(f"duplicate candidate name=<{candidate.name}>")
            seen.add(candidate.name)

    @staticmethod
    def _model_id(model: Model) -> str | None:
        config = getattr(model, "config", None)
        if config is None:
            try:
                config = model.get_config()
            except Exception:
                config = None
        if isinstance(config, dict):
            return config.get("model_id")
        return getattr(config, "model_id", None)
