"""
Data models for the Bioresearch Environment.

Biological reasoning environment for training and evaluating AI agents
on real-world genomics and proteomics tasks using GRPO-compatible
reward design.

The schema supports two modes:
    - Legacy single-step tasks (dna_classification, dna_reasoning,
      evidence_ranking, protein_function) — the agent always submits
      a terminal action.
    - New long-horizon lab tasks (target_discovery_lab,
      protein_hypothesis_lab, curriculum_self_play) — the agent can
      either invoke a tool (``tool_name`` + ``tool_args``) or submit
      a terminal answer (``submit=True``).
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class BioresearchAction(Action):
    """Action submitted by the agent for a bioresearch task."""

    task_id: str = Field(..., description="ID of the task instance being answered")
    answer: str = Field(default="", description="Disease name or function description (used on submit)")
    reasoning: Optional[str] = Field(default=None, description="Biological reasoning chain")
    go_terms: Optional[List[str]] = Field(default=None, description="Predicted GO terms")
    subcellular_location: Optional[str] = Field(default=None, description="Predicted subcellular location")
    ranked_diseases: Optional[List[str]] = Field(default=None, description="Ordered disease ranking, most likely first")
    elimination_reasoning: Optional[Dict[str, str]] = Field(default=None, description="Disease name -> why eliminated")

    # ── Long-horizon lab mode (new) ──────────────────────────────────────
    tool_name: Optional[str] = Field(default=None, description="Name of the tool to invoke (lab mode)")
    tool_args: Optional[Dict[str, Any]] = Field(default=None, description="Arguments passed to the tool")
    submit: bool = Field(default=False, description="If True the episode is finalised and graded")
    proposed_intervention: Optional[Dict[str, str]] = Field(
        default=None,
        description="Proposed therapeutic intervention, e.g. {'mode': 'inhibit', 'target': 'PDE11A'}",
    )

    # ── v2 task fields (clinical diagnosis, perturbation QA, ligand design) ──
    predicted_ligand: Optional[str] = Field(
        default=None,
        description="Predicted drug (SMILES string or named drug) for ligand_design and DRUG_DESIGN phase",
    )
    perturbation_answers: Optional[Dict[str, bool]] = Field(
        default=None,
        description="{pair_id: yes/no} predictions for a perturbation_qa batch",
    )
    differential_ranking: Optional[List[str]] = Field(
        default=None,
        description="Ordered differential diagnoses for clinical_diagnosis (most likely first)",
    )

    # ── v3 task fields (directional perturbation + benchmark umbrella) ──
    direction_answers: Optional[Dict[str, str]] = Field(
        default=None,
        description='{pair_id: "Increase" | "Decrease" | "Unknown"} predictions for perturbation_direction_qa / perturbation_benchmark',
    )
    mentioned_genes: Optional[List[str]] = Field(
        default=None,
        description="Explicit gene-symbol list for kegg_pathway_reasoning pathway coverage scoring",
    )


class BioresearchObservation(Observation):
    """Observation returned by the environment for a bioresearch task."""

    task_id: str = Field(default="", description="Unique ID for this problem instance")
    task_type: str = Field(default="", description="Task identifier (legacy or lab task type)")
    question: str = Field(default="", description="The question/prompt for the agent")
    sequence_data: Dict[str, str] = Field(default_factory=dict, description="Sequence data (DNA ref/var or protein)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Pathway info, gene list, organism, etc.")
    candidate_diseases: Optional[List[str]] = Field(
        default=None, description="4 candidate diseases for the evidence_ranking task"
    )

    # ── Long-horizon lab mode (new) ──────────────────────────────────────
    phase: str = Field(default="", description="Current lab phase: TARGET | CHARACTERIZE | HYPOTHESIZE | INTERVENE | SUBMIT")
    tool_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Response to the most recent tool call (None on reset)"
    )
    remaining_steps: int = Field(default=0, description="Maximum number of additional steps before forced submit")
    notebook: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rolling evidence log accumulated from prior tool calls (each entry capped)",
    )
    available_tools: List[str] = Field(
        default_factory=list, description="Tool names currently available to the agent"
    )

    # ── v2 task fields ───────────────────────────────────────────────────
    ligand_candidates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Candidate ligand rows from get_candidate_ligands (ligand_design observations)",
    )
    perturbation_batch: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of {pair_id, question, query_gene, target_gene, cell_line} for perturbation_qa",
    )
    differentials: Optional[List[str]] = Field(
        default=None,
        description="Reference differential diagnosis candidates shown to the agent in clinical tasks",
    )

    # ── v3 task fields ───────────────────────────────────────────────────
    pathway_graph: Optional[str] = Field(
        default=None,
        description="Raw KEGG-style pathway graph string (e.g. 'TARDBP* -| CxI -> Q') for kegg_pathway_reasoning",
    )
    genes_in_pathway: Optional[List[str]] = Field(
        default=None,
        description="Parsed gene list from the KEGG pathway context for kegg_pathway_reasoning",
    )
    structure_path: Optional[str] = Field(
        default=None,
        description="AlphaFold structure filename hint (e.g. 'AF-Q13148-F1-model_v6.pdb') for lab tasks",
    )
    direction_batch: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Directional CRISPRi batch: list of {pair_id, query_gene, target_gene, cell_line, question} entries",
    )
    benchmark_variants: Optional[List[str]] = Field(
        default=None,
        description="Per-pair variant labels for perturbation_benchmark (pert_dir | pert_de | gse_pert | gse_gene)",
    )
