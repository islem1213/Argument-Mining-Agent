"""argmine — hybrid argument mining: discourse-aware heuristics, local NLI or LLM judges,
plus formal argumentation semantics on top of the mined graph."""
from .pipeline import ArgumentMiner, PipelineConfig, run
from .schema import ArgumentGraph, Component, ComponentLabel, Relation, RelationType, Stance

__version__ = "0.2.0"
__all__ = ["ArgumentGraph", "ArgumentMiner", "Component", "ComponentLabel", "PipelineConfig", "Relation",
           "RelationType", "Stance", "__version__", "run"]
