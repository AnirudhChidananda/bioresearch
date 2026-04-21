"""End-to-end tests for the Virtual Tumor Board multi-turn task."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import BioresearchAction
from server.bioresearch_environment import BioresearchEnvironment


class TestTumorBoardFlow:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_vtb_populates_multi_turn_fields(self):
        obs = self.env.reset(task_type="virtual_tumor_board", task_id="dna_007")
        assert obs.task_type == "virtual_tumor_board"
        assert obs.max_turns == 8
        assert obs.turn_count == 0
        assert obs.candidate_diseases is not None
        assert len(obs.candidate_diseases) == 4
        assert obs.available_tools is not None
        assert obs.available_specialists is not None
        assert "submit_consensus" in obs.available_tools
        assert not obs.done

    def test_ask_specialist_advances_turn_without_termination(self):
        self.env.reset(task_type="virtual_tumor_board", task_id="dna_007")
        r = self.env.step(BioresearchAction(
            task_id="dna_007",
            tool_name="ask_specialist",
            tool_args={"role": "geneticist", "question": "variant?"},
        ))
        assert not r.done
        assert r.turn_count == 1
        assert r.tool_output is not None
        assert "[geneticist]" in r.tool_output

    def test_submit_consensus_terminates(self):
        self.env.reset(task_type="virtual_tumor_board", task_id="dna_007")
        r = self.env.step(BioresearchAction(
            task_id="dna_007",
            tool_name="submit_consensus",
            tool_args={"answer": "creutzfeldt-jakob disease", "reasoning": "test"},
        ))
        assert r.done
        assert r.reward is not None
        assert r.reward > 0.0

    def test_max_turns_forces_termination(self):
        self.env.reset(task_type="virtual_tumor_board", task_id="dna_007", max_turns=3)
        # Make 3 tool calls without submitting
        for _ in range(3):
            r = self.env.step(BioresearchAction(
                task_id="dna_007",
                tool_name="blast_lookup",
                tool_args={},
            ))
        assert r.done
        bd = (r.metadata or {}).get("score_breakdown", {})
        assert bd.get("forced_termination") is True

    def test_invalid_tool_name_does_not_crash(self):
        self.env.reset(task_type="virtual_tumor_board", task_id="dna_007")
        r = self.env.step(BioresearchAction(
            task_id="dna_007",
            tool_name="delete_the_database",
            tool_args={},
        ))
        assert not r.done
        assert "ERROR" in (r.tool_output or "")


class TestTumorBoardRewardVariance:
    """Verify the consensus grader gives well-separated rewards for good vs bad rollouts."""

    def setup_method(self):
        self.env = BioresearchEnvironment()

    def _good_rollout(self, task_id: str, gold_answer: str) -> float:
        self.env.reset(task_type="virtual_tumor_board", task_id=task_id)
        self.env.step(BioresearchAction(
            task_id=task_id, tool_name="ask_specialist",
            tool_args={"role": "geneticist", "question": "variant?"},
        ))
        self.env.step(BioresearchAction(
            task_id=task_id, tool_name="ask_specialist",
            tool_args={"role": "pathway_analyst", "question": "pathway?"},
        ))
        self.env.step(BioresearchAction(
            task_id=task_id, tool_name="ask_specialist",
            tool_args={"role": "clinician", "question": "phenotype?"},
        ))
        r = self.env.step(BioresearchAction(
            task_id=task_id, tool_name="submit_consensus",
            tool_args={
                "answer": gold_answer,
                "reasoning": "Based on specialist consensus: the variant disrupts the pathway which activates downstream signaling leading to the phenotype. Step 1 activates Step 2 inhibits Step 3 triggers the disease.",
            },
        ))
        return r.reward

    def _bad_rollout(self, task_id: str) -> float:
        self.env.reset(task_type="virtual_tumor_board", task_id=task_id)
        r = self.env.step(BioresearchAction(
            task_id=task_id, tool_name="submit_consensus",
            tool_args={"answer": "diabetes mellitus type 2", "reasoning": "guessing"},
        ))
        return r.reward

    def test_good_rollout_beats_bad(self):
        from server.data_loader import DataLoader
        gold = DataLoader().get_dna_sample_by_id("dna_007").answer
        good = self._good_rollout("dna_007", gold)
        bad = self._bad_rollout("dna_007")
        assert good - bad >= 0.3, f"Insufficient spread: good={good}, bad={bad}"

    def test_grpo_replay_deterministic(self):
        """Two identical rollouts on the same task_id must give identical rewards."""
        from server.data_loader import DataLoader
        gold = DataLoader().get_dna_sample_by_id("dna_015").answer
        r1 = self._good_rollout("dna_015", gold)
        r2 = self._good_rollout("dna_015", gold)
        assert abs(r1 - r2) < 1e-9
