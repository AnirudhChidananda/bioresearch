"""
Data models for the Bioresearch Environment.

Biological reasoning environment for training and evaluating AI agents
on real-world genomics and proteomics tasks using GRPO-compatible
reward design. Supports both single-turn tasks (classification,
reasoning, ranking, protein function) and the multi-turn
Virtual Tumor Board task.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class BioresearchAction(Action):
    """Action submitted by the agent for a bioresearch task.

    Single-turn tasks use only `answer` + optional `reasoning`.
    Multi-turn tasks (virtual_tumor_board) use `tool_name` + `tool_args`
    each turn, and terminate with tool_name='submit_consensus'.
    """

    task_id: str = Field(..., description="ID of the task instance being answered")
    answer: str = Field(default="", description="Disease name (T1/T2/T4) or function description (T3)")
    reasoning: Optional[str] = Field(default=None, description="Biological reasoning chain")
    go_terms: Optional[List[str]] = Field(default=None, description="Predicted GO terms (protein_function only)")
    subcellular_location: Optional[str] = Field(default=None, description="Predicted subcellular location (protein_function only)")
    ranked_diseases: Optional[List[str]] = Field(default=None, description="Ordered disease ranking (evidence_ranking only)")
    elimination_reasoning: Optional[Dict[str, str]] = Field(default=None, description="Disease -> why eliminated (evidence_ranking only)")

    tool_name: Optional[str] = Field(
        default=None,
        description=(
            "For multi-turn tasks: tool to invoke this turn. One of "
            "'blast_lookup', 'pathway_expand', 'go_term_lookup', "
            "'literature_snippet', 'ask_specialist', 'submit_consensus'."
        ),
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments for the tool_name call (see server.tools).",
    )


class BioresearchObservation(Observation):
    """Observation returned by the environment for a bioresearch task."""

    task_id: str = Field(default="", description="Unique ID for this problem instance")
    task_type: str = Field(default="", description="Task identifier")
    question: str = Field(default="", description="The question/prompt for the agent")
    sequence_data: Dict[str, str] = Field(default_factory=dict, description="Sequence data (DNA ref/var or protein)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Pathway info, gene list, organism, etc.")
    candidate_diseases: Optional[List[str]] = Field(default=None, description="4 candidate diseases (evidence_ranking and virtual_tumor_board)")

    turn_count: int = Field(default=0, description="Current turn index in a multi-turn episode")
    max_turns: int = Field(default=1, description="Turn budget for the episode")
    tool_output: Optional[str] = Field(default=None, description="Textual output of the last tool call (multi-turn)")
    available_tools: Optional[List[str]] = Field(default=None, description="Names of tools the agent may invoke (multi-turn)")
    available_specialists: Optional[List[str]] = Field(default=None, description="Specialist roles the agent may ask_specialist (multi-turn)")
    history_summary: Optional[List[Dict[str, Any]]] = Field(default=None, description="Compact per-turn log of tool calls so far")
