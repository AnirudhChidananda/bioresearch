"""
Bioresearch Environment Implementation.

A biological reasoning environment for evaluating AI agents on
DNA mutation analysis, protein function prediction, variant
pathogenicity ranking, and multi-agent Virtual Tumor Board cases.
GRPO-compatible with deterministic same-prompt replay.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BioresearchAction, BioresearchObservation
except ImportError:
    from models import BioresearchAction, BioresearchObservation

try:
    from . import actors as actors_module
    from . import tools as tools_module
    from .data_loader import DataLoader, DNASample, ProteinSample
    from .graders import (
        grade_consensus,
        grade_dna_classification,
        grade_dna_reasoning,
        grade_evidence_ranking,
        grade_protein_function,
    )
except ImportError:
    from server import actors as actors_module
    from server import tools as tools_module
    from server.data_loader import DataLoader, DNASample, ProteinSample
    from server.graders import (
        grade_consensus,
        grade_dna_classification,
        grade_dna_reasoning,
        grade_evidence_ranking,
        grade_protein_function,
    )


SINGLE_TURN_TASKS = (
    "dna_classification",
    "dna_reasoning",
    "evidence_ranking",
    "protein_function",
)
MULTI_TURN_TASKS = ("virtual_tumor_board",)
TASK_TYPES = SINGLE_TURN_TASKS + MULTI_TURN_TASKS

MAX_SEQ_DISPLAY = 500
DEFAULT_MAX_TURNS = 8


def _truncate_sequence(seq: str, max_len: int = MAX_SEQ_DISPLAY) -> str:
    if len(seq) <= max_len * 2:
        return seq
    return seq[:max_len] + " [...] " + seq[-max_len:]


def _extract_pathway_genes(question: str) -> List[str]:
    import re
    genes: List[str] = []
    section = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    if section:
        for match in re.finditer(r"(\w+)\s*;", section.group(1)):
            genes.append(match.group(1))
    return genes


class BioresearchEnvironment(Environment):
    """
    Biological reasoning environment with 5 tasks of increasing difficulty.

    Single-turn tasks (terminate after one `step`):
        - dna_classification
        - dna_reasoning
        - evidence_ranking
        - protein_function

    Multi-turn task (terminates on submit_consensus or max_turns):
        - virtual_tumor_board

    Supports GRPO same-prompt replay via reset(task_id=...).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._data = DataLoader()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_type: str = ""
        self._current_task_id: str = ""
        self._current_gold_sample: Optional[Any] = None
        self._current_distractors: List[str] = []
        self._current_pathway_genes: List[str] = []

        # Multi-turn state
        self._turn_history: List[Dict[str, Any]] = []
        self._max_turns: int = DEFAULT_MAX_TURNS
        self._episode_done: bool = True

    # ─────────────────────────────────────────────────────────────────────
    # reset
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, **kwargs) -> BioresearchObservation:
        task_id = kwargs.get("task_id")
        task_type = kwargs.get("task_type")
        max_turns = int(kwargs.get("max_turns", DEFAULT_MAX_TURNS))

        if task_id:
            if task_id.startswith("dna_"):
                sample = self._data.get_dna_sample_by_id(task_id)
                if not task_type:
                    task_type = "dna_classification"
            elif task_id.startswith("protein_"):
                sample = self._data.get_protein_sample_by_id(task_id)
                if not task_type:
                    task_type = "protein_function"
            else:
                raise ValueError(f"Cannot determine task from task_id: {task_id}")
        else:
            if not task_type:
                task_type = random.choice(list(TASK_TYPES))

            if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking", "virtual_tumor_board"):
                task_id, sample = self._data.get_random_dna_sample()
            elif task_type == "protein_function":
                task_id, sample = self._data.get_random_protein_sample()
            else:
                raise ValueError(f"Unknown task_type: {task_type}")

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_type = task_type
        self._current_task_id = task_id
        self._current_gold_sample = sample
        self._current_distractors = []
        self._current_pathway_genes = []
        self._turn_history = []
        self._max_turns = max(1, max_turns) if task_type in MULTI_TURN_TASKS else 1
        self._episode_done = False

        return self._build_observation(task_type, task_id, sample)

    # ─────────────────────────────────────────────────────────────────────
    # Observation builders
    # ─────────────────────────────────────────────────────────────────────

    def _build_observation(self, task_type: str, task_id: str, sample) -> BioresearchObservation:
        if task_type == "virtual_tumor_board":
            return self._build_tumor_board_observation(task_id, sample)
        if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking"):
            return self._build_dna_observation(task_type, task_id, sample)
        if task_type == "protein_function":
            return self._build_protein_observation(task_id, sample)
        raise ValueError(f"Unknown task_type: {task_type}")

    def _build_dna_observation(self, task_type: str, task_id: str, sample: DNASample) -> BioresearchObservation:
        pathway_genes = _extract_pathway_genes(sample.question)
        self._current_pathway_genes = pathway_genes

        sequence_data = {
            "reference_sequence": _truncate_sequence(sample.reference_sequence),
            "variant_sequence": _truncate_sequence(sample.variant_sequence),
        }

        context: Dict[str, Any] = {
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
            turn_count=0,
            max_turns=1,
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
            turn_count=0,
            max_turns=1,
            done=False,
            reward=0.0,
        )

    def _build_tumor_board_observation(self, task_id: str, sample: DNASample) -> BioresearchObservation:
        pathway_genes = _extract_pathway_genes(sample.question)
        self._current_pathway_genes = pathway_genes
        self._current_distractors = self._data.get_distractors(task_id, n=3)
        candidates = self._data.get_candidate_diseases(task_id)

        sequence_data = {
            "reference_sequence": _truncate_sequence(sample.reference_sequence),
            "variant_sequence": _truncate_sequence(sample.variant_sequence),
        }

        context: Dict[str, Any] = {
            "pathway_genes": pathway_genes,
            "full_reference_sequence": sample.reference_sequence,
            "full_variant_sequence": sample.variant_sequence,
            "specialist_roles": actors_module.list_roles(),
            "role_descriptions": actors_module.ROLE_DESCRIPTIONS,
        }

        question = (
            f"VIRTUAL TUMOR BOARD — Case {task_id}\n\n"
            f"You are the ORCHESTRATOR of a multidisciplinary diagnostic panel. "
            f"Use the available tools over up to {self._max_turns} turns to reach a confident diagnosis.\n\n"
            f"CASE CONTEXT:\n{sample.question}\n\n"
            f"CANDIDATE DIAGNOSES (choose one):\n"
            + "\n".join(f"  - {c}" for c in candidates)
            + "\n\nAVAILABLE TOOLS:\n"
            + "\n".join(f"  - {t}" for t in sorted(tools_module.TOOL_NAMES))
            + "\n\nAVAILABLE SPECIALISTS (for ask_specialist):\n"
            + "\n".join(f"  - {r}: {d}" for r, d in actors_module.ROLE_DESCRIPTIONS.items())
            + "\n\nTo submit your final diagnosis, call submit_consensus with "
            f"{{'answer': <disease>, 'reasoning': <synthesised justification>}}."
        )

        return BioresearchObservation(
            task_id=task_id,
            task_type="virtual_tumor_board",
            question=question,
            sequence_data=sequence_data,
            context=context,
            candidate_diseases=candidates,
            turn_count=0,
            max_turns=self._max_turns,
            available_tools=sorted(tools_module.TOOL_NAMES),
            available_specialists=actors_module.list_roles(),
            history_summary=[],
            done=False,
            reward=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────
    # step
    # ─────────────────────────────────────────────────────────────────────

    def step(self, action: BioresearchAction) -> BioresearchObservation:  # type: ignore[override]
        self._state.step_count += 1
        task_type = self._current_task_type
        sample = self._current_gold_sample

        if sample is None or self._episode_done:
            return BioresearchObservation(
                task_id=self._current_task_id,
                task_type=task_type,
                question="Error: no active episode. Call reset() first.",
                sequence_data={},
                context={},
                done=True,
                reward=0.01,
            )

        if task_type in MULTI_TURN_TASKS:
            return self._step_multi_turn(action, sample)
        return self._step_single_turn(action, sample)

    # ─────────────────────────────────────────────────────────────────────
    # Single-turn step (legacy tasks)
    # ─────────────────────────────────────────────────────────────────────

    def _step_single_turn(self, action: BioresearchAction, sample) -> BioresearchObservation:
        task_type = self._current_task_type
        score, breakdown = self._grade_single_turn(task_type, action, sample)

        self._episode_done = True
        obs = BioresearchObservation(
            task_id=self._current_task_id,
            task_type=task_type,
            question="Episode complete.",
            sequence_data={},
            context={"score_breakdown": breakdown},
            turn_count=1,
            max_turns=1,
            done=True,
            reward=score,
            metadata={"score_breakdown": breakdown},
        )
        self._current_gold_sample = None
        return obs

    def _grade_single_turn(self, task_type: str, action: BioresearchAction, sample) -> tuple:
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

    # ─────────────────────────────────────────────────────────────────────
    # Multi-turn step (Virtual Tumor Board)
    # ─────────────────────────────────────────────────────────────────────

    def _step_multi_turn(self, action: BioresearchAction, sample) -> BioresearchObservation:
        tool_name = action.tool_name or ""
        tool_args = action.tool_args or {}

        # Backwards-compat: allow submit via `answer` + no tool_name on last turn.
        if not tool_name and action.answer:
            tool_name = "submit_consensus"
            tool_args = {"answer": action.answer, "reasoning": action.reasoning or ""}

        # Validate the tool call
        error = tools_module.validate_tool_call(tool_name, tool_args)
        turn_idx = len(self._turn_history) + 1

        # submit_consensus → grade & terminate
        if tool_name == "submit_consensus" and not error:
            final_answer = str(tool_args.get("answer", action.answer or ""))
            final_reasoning = tool_args.get("reasoning", action.reasoning)
            turn_record = {
                "turn": turn_idx,
                "tool": "submit_consensus",
                "args": {"answer": final_answer, "reasoning": final_reasoning},
                "output": "[episode terminated by orchestrator]",
            }
            self._turn_history.append(turn_record)
            return self._finalise_tumor_board(final_answer, final_reasoning, sample)

        # Invalid tool call → small penalty turn, no-op output
        if error:
            turn_record = {
                "turn": turn_idx,
                "tool": tool_name or "<empty>",
                "args": tool_args,
                "output": f"ERROR: {error}",
            }
            self._turn_history.append(turn_record)
            tool_output = turn_record["output"]
        else:
            try:
                tool_output = tools_module.dispatch(sample, tool_name, tool_args, actors_module)
            except Exception as exc:  # pragma: no cover
                tool_output = f"ERROR running {tool_name}: {exc}"

            turn_record = {
                "turn": turn_idx,
                "tool": tool_name,
                "args": tool_args,
                "output": tool_output,
            }
            self._turn_history.append(turn_record)

        # Out of budget? auto-finalise with the agent's best current guess (empty string triggers 0 answer score).
        if len(self._turn_history) >= self._max_turns:
            return self._finalise_tumor_board(action.answer or "", action.reasoning, sample, forced=True)

        return self._build_intermediate_observation(sample, tool_output)

    def _build_intermediate_observation(self, sample, last_output: str) -> BioresearchObservation:
        candidates = self._data.get_candidate_diseases(self._current_task_id)
        summary = [
            {
                "turn": t["turn"],
                "tool": t["tool"],
                "args": t.get("args", {}),
                "output_preview": (t.get("output", "") or "")[:200],
            }
            for t in self._turn_history
        ]

        return BioresearchObservation(
            task_id=self._current_task_id,
            task_type="virtual_tumor_board",
            question=(
                f"Turn {len(self._turn_history)}/{self._max_turns}. "
                f"Last tool output is provided. Decide your next tool call, or submit_consensus."
            ),
            sequence_data={},
            context={
                "pathway_genes": self._current_pathway_genes,
                "specialist_roles": actors_module.list_roles(),
            },
            candidate_diseases=candidates,
            turn_count=len(self._turn_history),
            max_turns=self._max_turns,
            tool_output=last_output,
            available_tools=sorted(tools_module.TOOL_NAMES),
            available_specialists=actors_module.list_roles(),
            history_summary=summary,
            done=False,
            reward=0.0,
        )

    def _finalise_tumor_board(
        self,
        final_answer: str,
        final_reasoning: Optional[str],
        sample: DNASample,
        forced: bool = False,
    ) -> BioresearchObservation:
        relevant = actors_module.relevant_specialists(sample)
        score, breakdown = grade_consensus(
            predicted_answer=final_answer,
            predicted_reasoning=final_reasoning,
            gold_answer=sample.answer,
            gold_reasoning=sample.reasoning,
            pathway_genes=self._current_pathway_genes,
            turn_history=self._turn_history,
            relevant_specialists=relevant,
            max_turns=self._max_turns,
        )
        breakdown["forced_termination"] = forced

        summary = [
            {
                "turn": t["turn"],
                "tool": t["tool"],
                "args": t.get("args", {}),
                "output_preview": (t.get("output", "") or "")[:200],
            }
            for t in self._turn_history
        ]

        self._episode_done = True
        obs = BioresearchObservation(
            task_id=self._current_task_id,
            task_type="virtual_tumor_board",
            question=(
                "Episode complete — forced termination (out of turns)."
                if forced else "Episode complete — consensus submitted."
            ),
            sequence_data={},
            context={"score_breakdown": breakdown},
            turn_count=len(self._turn_history),
            max_turns=self._max_turns,
            history_summary=summary,
            done=True,
            reward=score,
            metadata={"score_breakdown": breakdown, "turn_history": self._turn_history},
        )
        self._current_gold_sample = None
        return obs

    @property
    def state(self) -> State:
        return self._state
