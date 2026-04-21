"""Integration tests for the Bioresearch Environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import BioresearchAction, BioresearchObservation
from server.bioresearch_environment import BioresearchEnvironment


class TestEnvironmentReset:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_dna_classification(self):
        obs = self.env.reset(task_type="dna_classification")
        assert isinstance(obs, BioresearchObservation)
        assert obs.task_type == "dna_classification"
        assert obs.task_id.startswith("dna_")
        assert obs.question
        assert obs.sequence_data
        assert "reference_sequence" in obs.sequence_data
        assert obs.done is False

    def test_reset_dna_reasoning(self):
        obs = self.env.reset(task_type="dna_reasoning")
        assert obs.task_type == "dna_reasoning"
        assert obs.question

    def test_reset_evidence_ranking(self):
        obs = self.env.reset(task_type="evidence_ranking")
        assert obs.task_type == "evidence_ranking"
        assert obs.candidate_diseases is not None
        assert len(obs.candidate_diseases) == 4

    def test_reset_protein_function(self):
        obs = self.env.reset(task_type="protein_function")
        assert obs.task_type == "protein_function"
        assert obs.task_id.startswith("protein_")
        assert "sequence" in obs.sequence_data

    def test_reset_random_task(self):
        obs = self.env.reset()
        assert obs.task_type in (
            "dna_classification",
            "dna_reasoning",
            "evidence_ranking",
            "protein_function",
            "virtual_tumor_board",
        )


class TestGRPOSamePromptReplay:
    """Verify that reset(task_id=X) returns identical observations."""

    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_dna_replay(self):
        obs1 = self.env.reset(task_type="dna_classification", task_id="dna_000")
        obs2 = self.env.reset(task_type="dna_classification", task_id="dna_000")
        assert obs1.task_id == obs2.task_id
        assert obs1.question == obs2.question
        assert obs1.sequence_data == obs2.sequence_data

    def test_protein_replay(self):
        obs1 = self.env.reset(task_id="protein_000")
        obs2 = self.env.reset(task_id="protein_000")
        assert obs1.question == obs2.question
        assert obs1.sequence_data == obs2.sequence_data

    def test_evidence_ranking_replay(self):
        obs1 = self.env.reset(task_type="evidence_ranking", task_id="dna_005")
        obs2 = self.env.reset(task_type="evidence_ranking", task_id="dna_005")
        assert obs1.candidate_diseases == obs2.candidate_diseases


class TestEnvironmentStep:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_step_correct_classification(self):
        obs = self.env.reset(task_type="dna_classification", task_id="dna_000")
        from server.data_loader import DataLoader
        dl = DataLoader()
        gold = dl.get_dna_sample_by_id("dna_000")

        result = self.env.step(BioresearchAction(task_id="dna_000", answer=gold.answer))
        assert result.done is True
        assert result.reward >= 0.70

    def test_step_wrong_classification(self):
        self.env.reset(task_type="dna_classification", task_id="dna_000")
        result = self.env.step(BioresearchAction(task_id="dna_000", answer="completely wrong disease xyz"))
        assert result.done is True
        assert result.reward <= 0.30

    def test_step_protein_function(self):
        self.env.reset(task_type="protein_function", task_id="protein_000")
        result = self.env.step(BioresearchAction(
            task_id="protein_000",
            answer="ion channel protein",
            reasoning="Contains channel domains",
            subcellular_location="membrane",
            go_terms=["GO:0005886"],
        ))
        assert result.done is True
        assert 0.01 <= result.reward <= 0.99

    def test_step_evidence_ranking(self):
        self.env.reset(task_type="evidence_ranking", task_id="dna_000")
        from server.data_loader import DataLoader
        dl = DataLoader()
        gold = dl.get_dna_sample_by_id("dna_000")

        result = self.env.step(BioresearchAction(
            task_id="dna_000",
            answer=gold.answer,
            ranked_diseases=[gold.answer, "disease a", "disease b", "disease c"],
            elimination_reasoning={"disease a": "wrong pathway", "disease b": "wrong gene", "disease c": "unrelated"},
            reasoning="Step 1: The mutation affects the pathway. Step 2: This leads to disease.",
        ))
        assert result.done is True
        assert result.reward >= 0.20

    def test_step_without_reset_returns_error(self):
        env = BioresearchEnvironment()
        env.reset(task_type="dna_classification")
        env.step(BioresearchAction(task_id="x", answer="test"))
        result = env.step(BioresearchAction(task_id="x", answer="test again"))
        assert result.done is True
        assert result.reward == 0.01


class TestStateManagement:
    def test_state_episode_id_changes_on_reset(self):
        env = BioresearchEnvironment()
        env.reset(task_type="dna_classification")
        ep1 = env.state.episode_id
        env.reset(task_type="dna_classification")
        ep2 = env.state.episode_id
        assert ep1 != ep2

    def test_step_count_increments(self):
        env = BioresearchEnvironment()
        assert env.state.step_count == 0
        env.reset(task_type="dna_classification")
        assert env.state.step_count == 0
        env.step(BioresearchAction(task_id="x", answer="test"))
        assert env.state.step_count == 1
