"""
Data models for the Bioresearch Environment.

Biological reasoning environment for training and evaluating AI agents
on real-world genomics and proteomics tasks using GRPO-compatible
reward design.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class BioresearchAction(Action):
    """Action submitted by the agent for a bioresearch task."""

    task_id: str = Field(..., description="ID of the task instance being answered")
    answer: str = Field(..., description="Disease name (T1/T2/T4) or function description (T3)")
    reasoning: Optional[str] = Field(default=None, description="Biological reasoning chain (T2/T3/T4)")
    go_terms: Optional[List[str]] = Field(default=None, description="Predicted GO terms (T3 only)")
    subcellular_location: Optional[str] = Field(default=None, description="Predicted subcellular location (T3 only)")
    ranked_diseases: Optional[List[str]] = Field(default=None, description="Ordered disease ranking, most likely first (T4 only)")
    elimination_reasoning: Optional[Dict[str, str]] = Field(default=None, description="Disease name -> why eliminated (T4 only)")


class BioresearchObservation(Observation):
    """Observation returned by the environment for a bioresearch task."""

    task_id: str = Field(default="", description="Unique ID for this problem instance")
    task_type: str = Field(default="", description="dna_classification | dna_reasoning | protein_function | evidence_ranking")
    question: str = Field(default="", description="The question/prompt for the agent")
    sequence_data: Dict[str, str] = Field(default_factory=dict, description="Sequence data (DNA ref/var or protein)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Pathway info, gene list, organism, etc.")
    candidate_diseases: Optional[List[str]] = Field(default=None, description="4 candidate diseases for evidence_ranking task")
