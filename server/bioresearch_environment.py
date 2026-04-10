"""
Bioresearch Environment Implementation.

A biological reasoning environment for evaluating AI agents on
DNA mutation analysis, protein function prediction, and variant
pathogenicity ranking tasks. GRPO-compatible with deterministic
same-prompt replay.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BioresearchAction, BioresearchObservation
except ImportError:
    from models import BioresearchAction, BioresearchObservation

try:
    from .data_loader import DataLoader, DNASample, ProteinSample
    from .graders import (
        grade_dna_classification,
        grade_dna_reasoning,
        grade_evidence_ranking,
        grade_protein_function,
    )
except ImportError:
    from server.data_loader import DataLoader, DNASample, ProteinSample
    from server.graders import (
        grade_dna_classification,
        grade_dna_reasoning,
        grade_evidence_ranking,
        grade_protein_function,
    )

TASK_TYPES = ("dna_classification", "dna_reasoning", "protein_function", "evidence_ranking")
MAX_SEQ_DISPLAY = 500


def _truncate_sequence(seq: str, max_len: int = MAX_SEQ_DISPLAY) -> str:
    if len(seq) <= max_len * 2:
        return seq
    return seq[:max_len] + " [...] " + seq[-max_len:]


def _extract_pathway_genes(question: str) -> list:
    """Extract gene symbols from the question's pathway gene list."""
    import re
    genes = []
    section = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    if section:
        for match in re.finditer(r"(\w+)\s*;", section.group(1)):
            genes.append(match.group(1))
    return genes


class BioresearchEnvironment(Environment):
    """
    Biological reasoning environment with 4 tasks of increasing difficulty.
    Supports GRPO same-prompt replay via reset(task_id=...).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._data = DataLoader()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task_type: str = ""
        self._current_task_id: str = ""
        self._current_gold_sample = None
        self._current_distractors: list = []
        self._current_pathway_genes: list = []

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
                task_type = "protein_function"
            else:
                raise ValueError(f"Cannot determine task from task_id: {task_id}")
        else:
            if not task_type:
                task_type = random.choice(list(TASK_TYPES))

            if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking"):
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

        return self._build_observation(task_type, task_id, sample)

    def _build_observation(self, task_type: str, task_id: str, sample) -> BioresearchObservation:
        if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking"):
            return self._build_dna_observation(task_type, task_id, sample)
        elif task_type == "protein_function":
            return self._build_protein_observation(task_id, sample)
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
        sequence_data = {
            "sequence": sample.sequence,
        }
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

        score, breakdown = self._grade(task_type, action, sample)

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

    def _grade(self, task_type: str, action: BioresearchAction, sample) -> tuple:
        if task_type == "dna_classification":
            return grade_dna_classification(action.answer, sample.answer)

        elif task_type == "dna_reasoning":
            return grade_dna_reasoning(
                predicted_answer=action.answer,
                predicted_reasoning=action.reasoning or "",
                gold_answer=sample.answer,
                gold_reasoning=sample.reasoning,
                pathway_genes=self._current_pathway_genes,
            )

        elif task_type == "protein_function":
            return grade_protein_function(
                predicted_function=action.answer,
                predicted_location=action.subcellular_location,
                predicted_go_terms=action.go_terms,
                predicted_reasoning=action.reasoning,
                gold=sample,
            )

        elif task_type == "evidence_ranking":
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

    @property
    def state(self) -> State:
        return self._state
