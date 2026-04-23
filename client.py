"""Bioresearch Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import BioresearchAction, BioresearchObservation


class BioresearchEnv(
    EnvClient[BioresearchAction, BioresearchObservation, State]
):
    """
    Client for the Bioresearch Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Supports both legacy single-step tasks (dna_classification,
    dna_reasoning, evidence_ranking, protein_function, clinical_diagnosis,
    perturbation_qa) and the long-horizon lab tasks (target_discovery_lab,
    protein_hypothesis_lab, curriculum_self_play, clinical_diagnosis_lab,
    ligand_design) via the ``tool_name`` / ``tool_args`` / ``submit`` fields
    on :class:`BioresearchAction`.

    Example:
        >>> with BioresearchEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(task_type="dna_classification")
        ...     print(result.observation.question)
        ...     result = env.step(BioresearchAction(task_id="dna_000", answer="cushing syndrome"))
        ...     print(result.reward)
    """

    def _step_payload(self, action: BioresearchAction) -> Dict:
        payload: Dict[str, Any] = {
            "task_id": action.task_id,
            "answer": action.answer,
        }
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        if action.go_terms is not None:
            payload["go_terms"] = action.go_terms
        if action.subcellular_location is not None:
            payload["subcellular_location"] = action.subcellular_location
        if action.ranked_diseases is not None:
            payload["ranked_diseases"] = action.ranked_diseases
        if action.elimination_reasoning is not None:
            payload["elimination_reasoning"] = action.elimination_reasoning
        # Lab-mode fields
        if action.tool_name is not None:
            payload["tool_name"] = action.tool_name
        if action.tool_args is not None:
            payload["tool_args"] = action.tool_args
        if action.submit:
            payload["submit"] = True
        if action.proposed_intervention is not None:
            payload["proposed_intervention"] = action.proposed_intervention
        # v2 task fields
        if action.predicted_ligand is not None:
            payload["predicted_ligand"] = action.predicted_ligand
        if action.perturbation_answers is not None:
            payload["perturbation_answers"] = action.perturbation_answers
        if action.differential_ranking is not None:
            payload["differential_ranking"] = action.differential_ranking
        # v3 task fields
        if action.direction_answers is not None:
            payload["direction_answers"] = action.direction_answers
        if action.mentioned_genes is not None:
            payload["mentioned_genes"] = action.mentioned_genes
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[BioresearchObservation]:
        obs_data = payload.get("observation", {}) or {}
        observation = BioresearchObservation(
            task_id=obs_data.get("task_id", "") or "",
            task_type=obs_data.get("task_type", "") or "",
            question=obs_data.get("question", "") or "",
            sequence_data=obs_data.get("sequence_data") or {},
            context=obs_data.get("context") or {},
            candidate_diseases=obs_data.get("candidate_diseases"),
            phase=obs_data.get("phase", "") or "",
            tool_result=obs_data.get("tool_result"),
            remaining_steps=obs_data.get("remaining_steps", 0) or 0,
            notebook=obs_data.get("notebook") or [],
            available_tools=obs_data.get("available_tools") or [],
            ligand_candidates=obs_data.get("ligand_candidates"),
            perturbation_batch=obs_data.get("perturbation_batch"),
            differentials=obs_data.get("differentials"),
            pathway_graph=obs_data.get("pathway_graph"),
            genes_in_pathway=obs_data.get("genes_in_pathway"),
            structure_path=obs_data.get("structure_path"),
            direction_batch=obs_data.get("direction_batch"),
            benchmark_variants=obs_data.get("benchmark_variants"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}) or {},
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
