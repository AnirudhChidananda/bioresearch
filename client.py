"""Bioresearch Environment Client."""

from typing import Any, Dict, List, Optional

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
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[BioresearchObservation]:
        obs_data = payload.get("observation", {})
        observation = BioresearchObservation(
            task_id=obs_data.get("task_id", ""),
            task_type=obs_data.get("task_type", ""),
            question=obs_data.get("question", ""),
            sequence_data=obs_data.get("sequence_data", {}),
            context=obs_data.get("context", {}),
            candidate_diseases=obs_data.get("candidate_diseases"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
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
