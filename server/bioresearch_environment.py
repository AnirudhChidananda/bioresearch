"""
Bioresearch Environment Implementation.

The environment exposes two families of tasks:

Legacy single-step tasks (fast evaluators, used in tests & the GRPO
baseline leaderboard):

    - ``dna_classification``   — mutation → disease name.
    - ``dna_reasoning``        — mutation → disease name + stepwise mechanism.
    - ``evidence_ranking``     — rank 4 candidate diseases with elimination
                                 reasoning.
    - ``protein_function``     — protein sequence/domain → function + location + GO.

New long-horizon "Drug Discovery Lab" tasks (hackathon headline, designed
for Theme 3.1 *World Modeling* and Theme 2 *Long-Horizon Planning*):

    - ``target_discovery_lab``  — multi-step tool-calling investigation of
                                  a disease-linked pathway, terminating in
                                  a disease + mechanism + GO + intervention
                                  submission.
    - ``protein_hypothesis_lab``— protein investigation with a dense
                                  per-step process reward taken from the
                                  gold ``reasoning`` chain.
    - ``curriculum_self_play``  — bootstrap from catalogue ``<think>`` traces
                                  with progressive hint-hiding (Theme 4 bonus).

GRPO compatibility is preserved:

    * ``reset(task_id=...)`` is deterministic.
    * Every tool call is a pure function of ``(task_id, tool_name, args)``.
    * Every per-step shaping signal is deterministic.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BioresearchAction, BioresearchObservation
except ImportError:
    from models import BioresearchAction, BioresearchObservation

try:
    from .data_loader import (
        CatalogueSample,
        DataLoader,
        DNASample,
        ProteinSample,
        _extract_pathway_genes,
    )
    from .graders import (
        grade_dna_classification,
        grade_dna_reasoning,
        grade_evidence_ranking,
        grade_intervention,
        grade_leaf_go_f1,
        grade_process_trace,
        grade_protein_function,
        grade_tool_efficiency,
        _extract_steps,
    )
except ImportError:
    from server.data_loader import (
        CatalogueSample,
        DataLoader,
        DNASample,
        ProteinSample,
        _extract_pathway_genes,
    )
    from server.graders import (
        grade_dna_classification,
        grade_dna_reasoning,
        grade_evidence_ranking,
        grade_intervention,
        grade_leaf_go_f1,
        grade_process_trace,
        grade_protein_function,
        grade_tool_efficiency,
        _extract_steps,
    )


LEGACY_TASK_TYPES = (
    "dna_classification",
    "dna_reasoning",
    "protein_function",
    "evidence_ranking",
)
LAB_TASK_TYPES = (
    "target_discovery_lab",
    "protein_hypothesis_lab",
    "curriculum_self_play",
)
ALL_TASK_TYPES = LEGACY_TASK_TYPES + LAB_TASK_TYPES

MAX_SEQ_DISPLAY = 500
MAX_LAB_STEPS = 20

# Per-step shaping reward constants (kept small; main signal is terminal).
STEP_REWARD_TOOL_OK = 0.015
STEP_REWARD_TOOL_REDUNDANT = -0.010
STEP_REWARD_TOOL_ERROR = -0.005
STEP_REWARD_PROCESS_COEFF = 0.030  # scales process-trace similarity for per-step
STEP_REWARD_IDLE = -0.002


# -- Phase planner --------------------------------------------------------


_PHASES: List[Tuple[str, int]] = [
    ("TARGET", 3),
    ("CHARACTERIZE", 5),
    ("HYPOTHESIZE", 7),
    ("INTERVENE", 5),
]


def _phase_for_step(step_count: int) -> str:
    cumulative = 0
    for name, length in _PHASES:
        cumulative += length
        if step_count < cumulative:
            return name
    return "SUBMIT"


# -- Lab episode state ----------------------------------------------------


@dataclass
class _LabEpisode:
    """Mutable per-episode state for a lab-mode run."""

    task_type: str
    task_id: str
    max_steps: int = MAX_LAB_STEPS
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    notebook: List[Dict[str, Any]] = field(default_factory=list)
    seen_calls: set = field(default_factory=set)
    active: bool = True
    hidden_hints: List[str] = field(default_factory=list)  # for self-play curriculum


# -- Helpers --------------------------------------------------------------


def _truncate_sequence(seq: str, max_len: int = MAX_SEQ_DISPLAY) -> str:
    if len(seq) <= max_len * 2:
        return seq
    return seq[:max_len] + " [...] " + seq[-max_len:]


def _mutated_genes(question: str) -> List[str]:
    """Return genes marked with ``*`` in the Network Definition string."""
    network = re.search(r"Network Definition[^\n]*", question)
    if not network:
        return []
    return re.findall(r"\b([A-Z][A-Z0-9]{1,9})\*", network.group(0))


def _match_protein_for_gene(data: DataLoader, gene: str) -> Optional[ProteinSample]:
    """Find a ProteinSample whose names mention ``gene`` (case-insensitive)."""
    needle = gene.lower().strip()
    if not needle:
        return None
    for ids in data.get_all_protein_ids(baseline_only=False) + data.get_all_protein_ids(baseline_only=True):
        try:
            sample = data.get_protein_sample_by_id(ids)
        except KeyError:
            continue
        if needle in sample.protein_names.lower() or needle in sample.protein_id.lower():
            return sample
    return None


# =========================================================================
# Environment
# =========================================================================


class BioresearchEnvironment(Environment):
    """Biological reasoning environment — legacy tasks + long-horizon lab."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    LAB_TOOLS: Tuple[str, ...] = (
        "get_interpro",
        "get_ppi",
        "get_go",
        "get_sequence",
        "get_pathway",
        "get_subcellular_location",
        "search_catalogue",
    )

    def __init__(self):
        self._data = DataLoader()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_type: str = ""
        self._current_task_id: str = ""
        self._current_gold_sample: Any = None
        self._current_distractors: List[str] = []
        self._current_pathway_genes: List[str] = []
        self._lab: Optional[_LabEpisode] = None

    # -- Reset -------------------------------------------------------------

    def reset(self, **kwargs) -> BioresearchObservation:
        task_id = kwargs.get("task_id")
        task_type = kwargs.get("task_type")

        if task_id:
            if task_id.startswith("dna_"):
                sample = self._data.get_dna_sample_by_id(task_id)
                if not task_type:
                    task_type = "dna_classification"
            elif task_id.startswith("protein_"):
                sample = self._data.get_protein_sample_by_id(task_id)
                if not task_type:
                    task_type = "protein_function"
            elif task_id.startswith("catalogue_"):
                sample = self._data.get_catalogue_sample_by_id(task_id)
                if not task_type:
                    task_type = "curriculum_self_play"
            else:
                raise ValueError(f"Cannot determine task from task_id: {task_id}")
        else:
            if not task_type:
                task_type = random.choice(list(LEGACY_TASK_TYPES))

            if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking", "target_discovery_lab"):
                task_id, sample = self._data.get_random_dna_sample()
            elif task_type in ("protein_function", "protein_hypothesis_lab"):
                task_id, sample = self._data.get_random_protein_sample()
            elif task_type == "curriculum_self_play":
                task_id, sample = self._data.get_random_catalogue_sample()
            else:
                raise ValueError(f"Unknown task_type: {task_type}")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_type = task_type
        self._current_task_id = task_id
        self._current_gold_sample = sample
        self._current_distractors = []
        self._current_pathway_genes = []

        if task_type in LAB_TASK_TYPES:
            self._lab = _LabEpisode(task_type=task_type, task_id=task_id)
            if task_type == "curriculum_self_play":
                self._lab.hidden_hints = self._pick_curriculum_hints(task_id)
        else:
            self._lab = None

        return self._build_observation(task_type, task_id, sample)

    # -- Observation builders ---------------------------------------------

    def _build_observation(self, task_type: str, task_id: str, sample) -> BioresearchObservation:
        if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking"):
            return self._build_dna_observation(task_type, task_id, sample)
        if task_type == "protein_function":
            return self._build_protein_observation(task_id, sample)
        if task_type == "target_discovery_lab":
            return self._build_target_lab_observation(task_id, sample)
        if task_type == "protein_hypothesis_lab":
            return self._build_protein_lab_observation(task_id, sample)
        if task_type == "curriculum_self_play":
            return self._build_selfplay_observation(task_id, sample)
        raise ValueError(f"Unknown task_type: {task_type}")

    def _build_dna_observation(self, task_type: str, task_id: str, sample: DNASample) -> BioresearchObservation:
        pathway_genes = _extract_pathway_genes(sample.question)
        self._current_pathway_genes = pathway_genes

        sequence_data = {
            "reference_sequence": _truncate_sequence(sample.reference_sequence),
            "variant_sequence": _truncate_sequence(sample.variant_sequence),
        }
        context = {
            "pathway_genes": pathway_genes,
            "full_reference_sequence": sample.reference_sequence,
            "full_variant_sequence": sample.variant_sequence,
        }
        candidate_diseases = None
        if task_type == "evidence_ranking":
            candidate_diseases = self._data.get_candidate_diseases(task_id)
            self._current_distractors = self._data.get_distractors(task_id, n=3)

        return BioresearchObservation(
            task_id=task_id,
            task_type=task_type,
            question=sample.question,
            sequence_data=sequence_data,
            context=context,
            candidate_diseases=candidate_diseases,
            done=False,
            reward=0.0,
        )

    def _build_protein_observation(self, task_id: str, sample: ProteinSample) -> BioresearchObservation:
        sequence_data = {"sequence": sample.sequence}
        context = {
            "protein_name": sample.protein_names,
            "organism": sample.organism,
            "sequence_length": sample.length,
            "interpro_domains": sample.interpro_formatted,
        }
        return BioresearchObservation(
            task_id=task_id,
            task_type="protein_function",
            question=(
                f"Protein: {sample.protein_names}\n"
                f"Organism: {sample.organism}\n"
                f"Sequence length: {int(sample.length)} amino acids\n"
                f"InterPro domains:\n{sample.interpro_formatted}\n\n"
                f"Given this protein sequence and metadata, predict:\n"
                f"1. The protein's biological function\n"
                f"2. Its subcellular location\n"
                f"3. Relevant Gene Ontology (GO) terms\n"
                f"4. Your reasoning based on domain analysis and sequence features"
            ),
            sequence_data=sequence_data,
            context=context,
            done=False,
            reward=0.0,
        )

    def _build_target_lab_observation(self, task_id: str, sample: DNASample) -> BioresearchObservation:
        pathway_genes = _extract_pathway_genes(sample.question)
        self._current_pathway_genes = pathway_genes
        mutated = _mutated_genes(sample.question)
        opening = (
            "You are the Principal Investigator in an aging / rare-disease lab.\n"
            f"Opening brief: A pathway on chromosome and its perturbed genes "
            f"(marked with *) are shown below. Your job is to identify the "
            f"implicated disease, propose the mechanism, enumerate leaf-level GO "
            f"terms for the key implicated protein, and propose a therapeutic "
            f"intervention.\n\n"
            f"Pathway context:\n{_extract_pathway_section(sample.question)}\n\n"
            f"Perturbed genes: {', '.join(mutated) if mutated else 'unspecified'}\n\n"
            "Available tools (call them with tool_name / tool_args):\n"
            "  get_pathway(task_id=<this task_id> | gene=SYMBOL)\n"
            "  get_interpro(protein_id)\n"
            "  get_ppi(protein_id)\n"
            "  get_go(protein_id, branch='leaf'|'bp'|'mf'|'cc')\n"
            "  get_sequence(protein_id, window=[start,end])\n"
            "  get_subcellular_location(protein_id)\n"
            "  search_catalogue(keyword)\n\n"
            "When ready, set submit=True with answer (disease), reasoning, "
            "go_terms (leaf GO IDs), proposed_intervention={mode, target}."
        )
        return BioresearchObservation(
            task_id=task_id,
            task_type="target_discovery_lab",
            question=opening,
            sequence_data={
                "reference_sequence": _truncate_sequence(sample.reference_sequence),
                "variant_sequence": _truncate_sequence(sample.variant_sequence),
            },
            context={
                "pathway_genes": pathway_genes,
                "perturbed_genes": mutated,
            },
            phase=_phase_for_step(0),
            tool_result=None,
            remaining_steps=MAX_LAB_STEPS,
            notebook=[],
            available_tools=list(self.LAB_TOOLS),
            done=False,
            reward=0.0,
        )

    def _build_protein_lab_observation(self, task_id: str, sample: ProteinSample) -> BioresearchObservation:
        opening = (
            f"You are a protein-function specialist. Target protein: {sample.protein_id} "
            f"({sample.protein_names}) in {sample.organism}.\n\n"
            f"Length: {int(sample.length)} aa.\n"
            f"InterPro preview:\n{sample.interpro_formatted[:400]}\n\n"
            "Workflow: investigate with tools, then submit function description, "
            "subcellular location, leaf-level GO terms, and a step-by-step reasoning "
            "chain grounded in the evidence you gathered.\n\n"
            "Available tools (same schema as target_discovery_lab):\n"
            "  get_interpro(protein_id), get_ppi(protein_id), get_go(protein_id, branch=...)\n"
            "  get_sequence(protein_id, window=[start,end]), get_subcellular_location(protein_id),\n"
            "  search_catalogue(keyword)."
        )
        return BioresearchObservation(
            task_id=task_id,
            task_type="protein_hypothesis_lab",
            question=opening,
            sequence_data={"sequence": sample.sequence},
            context={
                "protein_name": sample.protein_names,
                "protein_id": sample.protein_id,
                "organism": sample.organism,
            },
            phase=_phase_for_step(0),
            tool_result=None,
            remaining_steps=MAX_LAB_STEPS,
            notebook=[],
            available_tools=list(self.LAB_TOOLS),
            done=False,
            reward=0.0,
        )

    def _build_selfplay_observation(self, task_id: str, sample: CatalogueSample) -> BioresearchObservation:
        # Curriculum: reveal everything except the hints listed in hidden_hints.
        hidden = set(self._lab.hidden_hints) if self._lab else set()
        shown: Dict[str, Any] = {
            "protein_id": sample.protein_id,
            "organism": sample.organism,
            "length": len(sample.sequence),
        }
        if "sequence" not in hidden:
            shown["sequence_preview"] = sample.sequence[:400]

        opening = (
            "Self-play curriculum. Produce a chain-of-thought analysis of this "
            "protein (paragraph-per-step), then write a structured 'Functional "
            "Summary' answer. Difficulty has hidden the following evidence: "
            f"{sorted(hidden) if hidden else '(nothing hidden — easy level)'}.\n\n"
            f"Known facts: {shown}\n\n"
            "Tools available: get_interpro, get_ppi, get_go, get_sequence, "
            "get_subcellular_location, search_catalogue. To finish, set "
            "submit=True and provide reasoning (your think steps, one per paragraph) "
            "plus answer (final summary)."
        )
        return BioresearchObservation(
            task_id=task_id,
            task_type="curriculum_self_play",
            question=opening,
            sequence_data={"sequence": sample.sequence if "sequence" not in hidden else ""},
            context={"organism": sample.organism, "hidden_hints": sorted(hidden)},
            phase=_phase_for_step(0),
            tool_result=None,
            remaining_steps=MAX_LAB_STEPS,
            notebook=[],
            available_tools=list(self.LAB_TOOLS),
            done=False,
            reward=0.0,
        )

    def _pick_curriculum_hints(self, task_id: str) -> List[str]:
        """Deterministic hint-hiding schedule keyed by task_id."""
        import hashlib

        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16)
        level = seed % 4  # 0 easy -> 3 hard
        options = ["sequence", "interpro", "ppi", "go"]
        return options[:level]

    # -- Step --------------------------------------------------------------

    def step(self, action: BioresearchAction) -> BioresearchObservation:  # type: ignore[override]
        self._state.step_count += 1
        task_type = self._current_task_type
        sample = self._current_gold_sample

        if sample is None:
            return BioresearchObservation(
                task_id=self._current_task_id,
                task_type=task_type,
                question="Error: no active episode. Call reset() first.",
                sequence_data={},
                context={},
                done=True,
                reward=0.01,
            )

        if task_type in LAB_TASK_TYPES:
            return self._lab_step(action)

        return self._legacy_step(action, sample, task_type)

    # -- Legacy step (unchanged behavior for the 4 original tasks) --------

    def _legacy_step(
        self,
        action: BioresearchAction,
        sample,
        task_type: str,
    ) -> BioresearchObservation:
        score, breakdown = self._grade_legacy(task_type, action, sample)
        obs = BioresearchObservation(
            task_id=self._current_task_id,
            task_type=task_type,
            question="Episode complete.",
            sequence_data={},
            context={"score_breakdown": breakdown},
            done=True,
            reward=score,
            metadata={"score_breakdown": breakdown},
        )
        self._current_gold_sample = None
        return obs

    def _grade_legacy(self, task_type: str, action: BioresearchAction, sample) -> Tuple[float, Dict[str, Any]]:
        if task_type == "dna_classification":
            return grade_dna_classification(action.answer, sample.answer)
        if task_type == "dna_reasoning":
            return grade_dna_reasoning(
                predicted_answer=action.answer,
                predicted_reasoning=action.reasoning or "",
                gold_answer=sample.answer,
                gold_reasoning=sample.reasoning,
                pathway_genes=self._current_pathway_genes,
            )
        if task_type == "protein_function":
            return grade_protein_function(
                predicted_function=action.answer,
                predicted_location=action.subcellular_location,
                predicted_go_terms=action.go_terms,
                predicted_reasoning=action.reasoning,
                gold=sample,
            )
        if task_type == "evidence_ranking":
            return grade_evidence_ranking(
                selected_disease=action.answer,
                ranked_diseases=action.ranked_diseases,
                elimination_reasoning=action.elimination_reasoning,
                supporting_evidence=action.reasoning,
                gold_disease=sample.answer,
                gold_reasoning=sample.reasoning,
                distractors=self._current_distractors,
                pathway_genes=self._current_pathway_genes,
            )
        return 0.01, {"error": f"Unknown task_type: {task_type}"}

    # -- Lab step ---------------------------------------------------------

    def _lab_step(self, action: BioresearchAction) -> BioresearchObservation:
        assert self._lab is not None
        lab = self._lab
        sample = self._current_gold_sample
        task_type = self._current_task_type

        submit = action.submit or (self._state.step_count >= MAX_LAB_STEPS)

        if submit or action.tool_name is None:
            return self._finalise_lab(action, forced=not action.submit)

        # Tool-calling step
        tool_name = action.tool_name
        tool_args = action.tool_args or {}

        signature = tool_name + ":" + ",".join(f"{k}={v}" for k, v in sorted(tool_args.items()))
        redundant = signature in lab.seen_calls
        lab.seen_calls.add(signature)

        tool_result = self._data.tool_response(tool_name, tool_args)

        # For the self-play curriculum, hide tool outputs flagged in hidden_hints.
        if task_type == "curriculum_self_play" and lab.hidden_hints:
            if (tool_name == "get_interpro" and "interpro" in lab.hidden_hints):
                tool_result = {"error": "blocked by curriculum (interpro hidden at this level)"}
            if (tool_name == "get_ppi" and "ppi" in lab.hidden_hints):
                tool_result = {"error": "blocked by curriculum (ppi hidden at this level)"}
            if (tool_name == "get_go" and "go" in lab.hidden_hints):
                tool_result = {"error": "blocked by curriculum (go hidden at this level)"}
            if (tool_name == "get_sequence" and "sequence" in lab.hidden_hints):
                tool_result = {"error": "blocked by curriculum (sequence hidden at this level)"}

        is_error = "error" in tool_result

        lab.tool_calls.append({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": tool_result,
            "step": self._state.step_count,
        })
        lab.notebook.append({
            "step": self._state.step_count,
            "tool": tool_name,
            "args": tool_args,
            "result": tool_result,
        })
        if len(lab.notebook) > 30:
            lab.notebook = lab.notebook[-30:]

        if is_error:
            step_reward = STEP_REWARD_TOOL_ERROR
        elif redundant:
            step_reward = STEP_REWARD_TOOL_REDUNDANT
        else:
            step_reward = STEP_REWARD_TOOL_OK

        phase = _phase_for_step(self._state.step_count)
        remaining = max(0, MAX_LAB_STEPS - self._state.step_count)

        metadata: Dict[str, Any] = {
            "step_reward": round(step_reward, 4),
            "phase": phase,
            "tool_name": tool_name,
            "is_redundant": redundant,
            "is_error": is_error,
        }

        return BioresearchObservation(
            task_id=self._current_task_id,
            task_type=task_type,
            question=f"Continue your investigation. Phase: {phase}. Steps remaining: {remaining}.",
            sequence_data={},
            context={"phase": phase},
            phase=phase,
            tool_result=tool_result,
            remaining_steps=remaining,
            notebook=list(lab.notebook),
            available_tools=list(self.LAB_TOOLS),
            done=False,
            reward=step_reward,
            metadata=metadata,
        )

    def _finalise_lab(self, action: BioresearchAction, forced: bool = False) -> BioresearchObservation:
        assert self._lab is not None
        lab = self._lab
        sample = self._current_gold_sample
        task_type = self._current_task_type

        if task_type == "target_discovery_lab":
            score, breakdown = self._grade_target_lab(action, sample, lab)
        elif task_type == "protein_hypothesis_lab":
            score, breakdown = self._grade_protein_lab(action, sample, lab)
        else:
            score, breakdown = self._grade_selfplay(action, sample, lab)

        breakdown["forced_submit"] = forced
        breakdown["total_tool_calls"] = len(lab.tool_calls)

        obs = BioresearchObservation(
            task_id=self._current_task_id,
            task_type=task_type,
            question="Episode complete.",
            sequence_data={},
            context={"score_breakdown": breakdown},
            phase="SUBMIT",
            tool_result=None,
            remaining_steps=0,
            notebook=list(lab.notebook),
            available_tools=[],
            done=True,
            reward=score,
            metadata={"score_breakdown": breakdown},
        )

        self._lab = None
        self._current_gold_sample = None
        return obs

    # -- Lab graders (composite) ------------------------------------------

    def _grade_target_lab(
        self,
        action: BioresearchAction,
        sample: DNASample,
        lab: _LabEpisode,
    ) -> Tuple[float, Dict[str, Any]]:
        pathway_genes = self._current_pathway_genes or _extract_pathway_genes(sample.question)
        mutated = _mutated_genes(sample.question)

        # 1) Disease match (25%)
        disease_score, disease_bd = grade_dna_classification(action.answer or "", sample.answer)
        disease_component = disease_score * 0.25

        # 2) Mechanism reasoning (25%)
        dna_score, dna_bd = grade_dna_reasoning(
            predicted_answer=action.answer or "",
            predicted_reasoning=action.reasoning or "",
            gold_answer=sample.answer,
            gold_reasoning=sample.reasoning,
            pathway_genes=pathway_genes,
        )
        reasoning_component = dna_bd.get("reasoning_component", 0.0) * (0.25 / 0.60 if 0.60 else 0.0)
        reasoning_component = min(0.25, reasoning_component + 0.0)

        # 3) Leaf GO hit (15%)
        implicated_protein: Optional[ProteinSample] = None
        for gene in mutated + pathway_genes:
            implicated_protein = _match_protein_for_gene(self._data, gene)
            if implicated_protein is not None:
                break
        if implicated_protein is not None:
            gold_leaf = "\n".join(
                x for x in [
                    implicated_protein.go_bp_leaf,
                    implicated_protein.go_mf_leaf,
                    implicated_protein.go_cc_leaf,
                ] if x
            )
            go_score, go_bd = grade_leaf_go_f1(action.go_terms, gold_leaf)
        else:
            go_score, go_bd = 0.30, {"note": "no implicated protein found for leaf GO check"}
        go_component = go_score * 0.15

        # 4) Intervention (15%)
        if implicated_protein is not None:
            int_score, int_bd = grade_intervention(
                action.proposed_intervention, implicated_protein, pathway_genes
            )
        else:
            int_score, int_bd = 0.30, {"note": "no implicated protein found for intervention check"}
        intervention_component = int_score * 0.15

        # 5) Tool efficiency (10%)
        tool_score, tool_bd = grade_tool_efficiency(lab.tool_calls, action.reasoning or "")
        tool_component = tool_score * 0.10

        # 6) Trace coherence (10%): each reasoning claim should cite tool evidence
        trace_score = self._trace_coherence_score(action.reasoning or "", lab.tool_calls)
        trace_component = trace_score * 0.10

        total = (
            disease_component
            + reasoning_component
            + go_component
            + intervention_component
            + tool_component
            + trace_component
        )

        breakdown = {
            "disease": {"score": round(disease_score, 4), "component": round(disease_component, 4), **disease_bd},
            "reasoning": {"component": round(reasoning_component, 4), "dna_breakdown": dna_bd},
            "leaf_go": {"score": round(go_score, 4), "component": round(go_component, 4), **(go_bd if isinstance(go_bd, dict) else {})},
            "intervention": {"score": round(int_score, 4), "component": round(intervention_component, 4), **(int_bd if isinstance(int_bd, dict) else {})},
            "tool_efficiency": {"score": round(tool_score, 4), "component": round(tool_component, 4), **tool_bd},
            "trace_coherence": {"score": round(trace_score, 4), "component": round(trace_component, 4)},
            "implicated_protein_id": implicated_protein.protein_id if implicated_protein else None,
        }
        return max(0.01, min(0.99, total)), breakdown

    def _grade_protein_lab(
        self,
        action: BioresearchAction,
        sample: ProteinSample,
        lab: _LabEpisode,
    ) -> Tuple[float, Dict[str, Any]]:
        # Core function/location (0.35)
        pf_score, pf_bd = grade_protein_function(
            predicted_function=action.answer or "",
            predicted_location=action.subcellular_location,
            predicted_go_terms=action.go_terms,
            predicted_reasoning=action.reasoning,
            gold=sample,
        )
        pf_component = pf_score * 0.35

        # Leaf GO (0.25)
        gold_leaf = "\n".join(
            x for x in [sample.go_bp_leaf, sample.go_mf_leaf, sample.go_cc_leaf] if x
        )
        leaf_score, leaf_bd = grade_leaf_go_f1(action.go_terms, gold_leaf)
        leaf_component = leaf_score * 0.25

        # Process trace vs gold reasoning (0.25) — the dense signal
        gold_steps = _extract_steps(sample.reasoning or "")
        pred_steps = _extract_steps(action.reasoning or "")
        proc_score, proc_bd = grade_process_trace(pred_steps, gold_steps)
        proc_component = proc_score * 0.25

        # Tool efficiency (0.15)
        tool_score, tool_bd = grade_tool_efficiency(lab.tool_calls, action.reasoning or "")
        tool_component = tool_score * 0.15

        total = pf_component + leaf_component + proc_component + tool_component
        breakdown = {
            "protein_function": {"score": round(pf_score, 4), "component": round(pf_component, 4), **pf_bd},
            "leaf_go": {"score": round(leaf_score, 4), "component": round(leaf_component, 4), **leaf_bd},
            "process_trace": {"score": round(proc_score, 4), "component": round(proc_component, 4), **proc_bd},
            "tool_efficiency": {"score": round(tool_score, 4), "component": round(tool_component, 4), **tool_bd},
        }
        return max(0.01, min(0.99, total)), breakdown

    def _grade_selfplay(
        self,
        action: BioresearchAction,
        sample: CatalogueSample,
        lab: _LabEpisode,
    ) -> Tuple[float, Dict[str, Any]]:
        # 1) Process trace similarity vs gold <think> steps (0.60)
        pred_steps = _extract_steps(action.reasoning or "")
        proc_score, proc_bd = grade_process_trace(pred_steps, sample.think_steps)
        proc_component = proc_score * 0.60

        # 2) Final structured-answer similarity (0.30) — token jaccard on the
        #    agent's ``answer`` vs gold structured_answer.
        from .graders import _tokenise, _jaccard  # local import — avoid top-level cycle  # noqa: E501

        pred_tokens = _tokenise(action.answer or "")
        gold_tokens = _tokenise(sample.structured_answer or "")
        summary_jaccard = _jaccard(pred_tokens, gold_tokens) if pred_tokens and gold_tokens else 0.0
        summary_component = summary_jaccard * 0.30

        # 3) Tool efficiency (0.10), scaled down since tools are optional here.
        if lab.tool_calls:
            tool_score, tool_bd = grade_tool_efficiency(lab.tool_calls, action.reasoning or "")
        else:
            tool_score, tool_bd = 0.50, {"note": "no tool calls (ok for self-play)"}
        tool_component = tool_score * 0.10

        total = proc_component + summary_component + tool_component
        breakdown = {
            "process_trace": {"score": round(proc_score, 4), "component": round(proc_component, 4), **proc_bd},
            "summary_jaccard": round(summary_jaccard, 4),
            "summary_component": round(summary_component, 4),
            "tool_efficiency": {"score": round(tool_score, 4), "component": round(tool_component, 4), **tool_bd},
            "curriculum_level": len(lab.hidden_hints),
            "hidden_hints": list(lab.hidden_hints),
        }
        return max(0.01, min(0.99, total)), breakdown

    def _trace_coherence_score(self, reasoning: str, tool_calls: List[Dict[str, Any]]) -> float:
        """Fraction of tool calls whose payload tokens overlap the reasoning text."""
        if not reasoning or not tool_calls:
            return 0.30
        from .graders import _normalise

        r_tokens = set(_normalise(reasoning).split())
        grounded = 0
        for call in tool_calls:
            payload = " ".join(str(v) for v in call.get("result", {}).values() if isinstance(v, (str, int, float)))
            payload_tokens = set(_normalise(payload).split())
            distinctive = {t for t in payload_tokens if len(t) >= 4}
            if distinctive & r_tokens:
                grounded += 1
        return 0.05 + min(1.0, grounded / max(1, len(tool_calls))) * 0.90

    # -- Public ------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state


# -- Pathway section extraction (local helper to avoid circular import) --


def _extract_pathway_section(question: str) -> str:
    network = re.search(r"Network Definition[^\n]*", question)
    genes = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    parts = []
    if network:
        parts.append(network.group(0))
    if genes:
        parts.append("Genes: " + genes.group(1).strip())
    return "\n".join(parts)
