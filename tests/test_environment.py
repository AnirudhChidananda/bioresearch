"""Integration tests for the Bioresearch Environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import BioresearchAction, BioresearchObservation
from server.bioresearch_environment import BioresearchEnvironment, LEGACY_TASK_TYPES


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
        # reset() without a task_type draws uniformly from the legacy pool.
        # Assert against the LEGACY_TASK_TYPES constant so this test never
        # goes stale when new legacy tasks are added or the canonical order
        # is reshuffled.
        obs = self.env.reset()
        assert obs.task_type in LEGACY_TASK_TYPES


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


# ── Lab Mode (long-horizon tool-calling) ────────────────────────────────

class TestLabReset:
    """Verify the three new long-horizon tasks reset into phased lab observations."""

    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_target_discovery_lab(self):
        obs = self.env.reset(task_type="target_discovery_lab")
        assert obs.task_type == "target_discovery_lab"
        assert obs.phase == "TARGET"
        assert obs.remaining_steps > 0
        assert isinstance(obs.notebook, list)
        assert obs.available_tools
        assert obs.done is False

    def test_reset_protein_hypothesis_lab(self):
        obs = self.env.reset(task_type="protein_hypothesis_lab")
        assert obs.task_type == "protein_hypothesis_lab"
        assert obs.phase == "TARGET"
        assert obs.available_tools

    def test_reset_curriculum_self_play(self):
        obs = self.env.reset(task_type="curriculum_self_play")
        assert obs.task_type == "curriculum_self_play"
        assert obs.available_tools


class TestLabEpisodeLoop:
    """Integration test: tool-call → notebook grows → submit → terminal reward."""

    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_tool_call_populates_notebook(self):
        obs = self.env.reset(task_type="protein_hypothesis_lab")
        tool = obs.available_tools[0] if obs.available_tools else "get_interpro"

        obs2 = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            tool_name=tool,
            tool_args={"gene": "TP53"},
        ))
        assert obs2.done is False
        assert obs2.phase in ("TARGET", "CHARACTERIZE", "HYPOTHESIZE", "INTERVENE", "SUBMIT")
        assert obs2.remaining_steps < obs.remaining_steps
        assert obs2.tool_result is not None
        assert len(obs2.notebook) == 1

    def test_submit_ends_episode_with_breakdown(self):
        obs = self.env.reset(task_type="protein_hypothesis_lab")

        self.env.step(BioresearchAction(
            task_id=obs.task_id,
            tool_name="get_interpro",
            tool_args={"protein_id": "P00533"},
        ))

        submit_obs = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            submit=True,
            answer="unknown function",
            reasoning="Insufficient evidence to fully characterise this protein.",
            go_terms=["GO:0003674"],
            proposed_intervention={"mode": "inhibit", "target": "TP53"},
        ))
        assert submit_obs.done is True
        assert 0.01 <= submit_obs.reward <= 0.99
        breakdown = submit_obs.metadata.get("score_breakdown") if submit_obs.metadata else None
        assert breakdown is not None
        assert isinstance(breakdown, dict)

    def test_target_discovery_lab_full_episode(self):
        obs = self.env.reset(task_type="target_discovery_lab")

        # Two tool calls then submit.
        self.env.step(BioresearchAction(
            task_id=obs.task_id,
            tool_name="get_pathway",
            tool_args={"gene": "TP53"},
        ))
        self.env.step(BioresearchAction(
            task_id=obs.task_id,
            tool_name="get_ppi",
            tool_args={"gene": "TP53"},
        ))
        final = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            submit=True,
            answer="cancer",
            reasoning="TP53 is a tumour suppressor; loss leads to uncontrolled proliferation.",
            proposed_intervention={"mode": "activate", "target": "TP53"},
        ))
        assert final.done is True
        assert 0.01 <= final.reward <= 0.99

    def test_lab_replay_deterministic(self):
        obs1 = self.env.reset(task_type="protein_hypothesis_lab", task_id=None)
        task_id = obs1.task_id
        obs2 = self.env.reset(task_type="protein_hypothesis_lab", task_id=task_id)
        assert obs1.task_id == obs2.task_id
        assert obs1.question == obs2.question
        assert obs1.phase == obs2.phase


# ── v2: New tasks ─────────────────────────────────────────────────────────


class TestClinicalDiagnosisReset:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_legacy(self):
        obs = self.env.reset(task_type="clinical_diagnosis")
        assert obs.task_type == "clinical_diagnosis"
        assert obs.task_id.startswith("diagnosis_")
        assert obs.question
        assert obs.differentials is not None
        assert len(obs.differentials) >= 2
        assert obs.done is False

    def test_reset_lab(self):
        obs = self.env.reset(task_type="clinical_diagnosis_lab")
        assert obs.task_type == "clinical_diagnosis_lab"
        assert obs.phase == "TARGET"
        assert obs.available_tools
        assert obs.done is False


class TestPerturbationQAReset:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_returns_batch(self):
        obs = self.env.reset(task_type="perturbation_qa")
        assert obs.task_type == "perturbation_qa"
        assert obs.perturbation_batch is not None
        assert len(obs.perturbation_batch) >= 1
        first = obs.perturbation_batch[0]
        for key in ("pair_id", "query_gene", "target_gene", "cell_line"):
            assert key in first

    def test_deterministic_replay(self):
        obs1 = self.env.reset(task_type="perturbation_qa", task_id="pertbatch_003")
        obs2 = self.env.reset(task_type="perturbation_qa", task_id="pertbatch_003")
        assert obs1.task_id == obs2.task_id
        assert obs1.perturbation_batch == obs2.perturbation_batch

    def test_step_grades_batch(self):
        obs = self.env.reset(task_type="perturbation_qa", task_id="pertbatch_000")
        answers = {p["pair_id"]: True for p in obs.perturbation_batch}
        result = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            perturbation_answers=answers,
        ))
        assert result.done is True
        assert 0.01 <= result.reward <= 0.99


class TestLigandDesignReset:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset(self):
        obs = self.env.reset(task_type="ligand_design")
        assert obs.task_type == "ligand_design"
        assert obs.task_id.startswith("ligand_")
        assert obs.question
        assert obs.ligand_candidates is not None

    def test_submit_ligand_scores(self):
        obs = self.env.reset(task_type="ligand_design")
        candidate = None
        if obs.ligand_candidates:
            candidate = obs.ligand_candidates[0].get("smiles")
        result = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            submit=True,
            predicted_ligand=candidate or "CCO",
            reasoning="Selected first high-pIC50 candidate.",
        ))
        assert result.done is True
        assert 0.01 <= result.reward <= 0.99


class TestClinicalDiagnosisStep:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_legacy_submission(self):
        obs = self.env.reset(task_type="clinical_diagnosis")
        result = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            answer=obs.differentials[0] if obs.differentials else "unknown",
            differential_ranking=obs.differentials,
            reasoning="Step 1: Considered key features. Step 2: Selected most likely diagnosis.",
        ))
        assert result.done is True
        assert 0.01 <= result.reward <= 0.99

    def test_lab_full_episode(self):
        obs = self.env.reset(task_type="clinical_diagnosis_lab")
        if obs.available_tools:
            self.env.step(BioresearchAction(
                task_id=obs.task_id,
                tool_name=obs.available_tools[0],
                tool_args={"gene": "TP53"},
            ))
        final = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            submit=True,
            answer=(obs.differentials[0] if obs.differentials else "unknown"),
            differential_ranking=obs.differentials,
            reasoning="Clinical workup points to the leading differential.",
        ))
        assert final.done is True
        assert 0.01 <= final.reward <= 0.99


class TestDrugDesignPhaseInHostLab:
    """target_discovery_lab now schedules a DRUG_DESIGN phase before SUBMIT."""

    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_predicted_ligand_folds_into_reward(self):
        obs = self.env.reset(task_type="target_discovery_lab")
        final = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            submit=True,
            answer="cancer",
            reasoning="Known tumour-suppressor loss.",
            proposed_intervention={"mode": "activate", "target": "TP53"},
            predicted_ligand="CCO",
        ))
        assert final.done is True
        assert 0.01 <= final.reward <= 0.99
        breakdown = final.metadata.get("score_breakdown") if final.metadata else None
        assert breakdown is not None


# ── v3: KEGG pathway reasoning task ─────────────────────────────────────

class TestKeggPathwayReasoningTask:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_exposes_pathway_graph(self):
        obs = self.env.reset(task_type="kegg_pathway_reasoning")
        assert obs.task_type == "kegg_pathway_reasoning"
        assert obs.task_id.startswith("kegg_")
        assert obs.pathway_graph
        assert obs.genes_in_pathway and len(obs.genes_in_pathway) >= 1
        assert obs.done is False

    def test_submission_scores(self):
        obs = self.env.reset(task_type="kegg_pathway_reasoning")
        result = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            answer="amyotrophic lateral sclerosis",
            reasoning=(
                "Step 1: " + (obs.pathway_graph or "")[:60] + ". "
                "Step 2: downstream effects cause the phenotype. "
                "Step 3: matches the ALS clinical picture."
            ),
            mentioned_genes=list(obs.genes_in_pathway or [])[:3],
        ))
        assert result.done is True
        assert 0.01 <= result.reward <= 0.99

    def test_deterministic_replay(self):
        obs1 = self.env.reset(task_id="kegg_0001")
        obs2 = self.env.reset(task_id="kegg_0001")
        assert obs1.pathway_graph == obs2.pathway_graph
        assert obs1.genes_in_pathway == obs2.genes_in_pathway


# ── v3: Perturbation direction QA ───────────────────────────────────────

class TestPerturbationDirectionTask:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_returns_direction_batch(self):
        obs = self.env.reset(task_type="perturbation_direction_qa")
        assert obs.task_type == "perturbation_direction_qa"
        assert obs.task_id.startswith("pertdir_")
        assert obs.direction_batch and len(obs.direction_batch) > 0
        for p in obs.direction_batch:
            assert "pair_id" in p
            assert "query_gene" in p
            assert "target_gene" in p

    def test_submission_scores(self):
        obs = self.env.reset(task_id="pertdir_000")
        answers = {p["pair_id"]: "Increase" for p in (obs.direction_batch or [])}
        result = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            direction_answers=answers,
        ))
        assert result.done is True
        assert 0.01 <= result.reward <= 0.99

    def test_deterministic_replay(self):
        obs1 = self.env.reset(task_id="pertdir_007")
        obs2 = self.env.reset(task_id="pertdir_007")
        ids1 = [p["pair_id"] for p in (obs1.direction_batch or [])]
        ids2 = [p["pair_id"] for p in (obs2.direction_batch or [])]
        assert ids1 == ids2
        assert len(ids1) > 0


# ── v3: Perturbation benchmark umbrella ─────────────────────────────────

class TestPerturbationBenchmarkTask:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_reset_spans_variants(self):
        obs = self.env.reset(task_type="perturbation_benchmark")
        assert obs.task_type == "perturbation_benchmark"
        assert obs.task_id.startswith("pertbench_")
        assert obs.benchmark_variants and len(obs.benchmark_variants) >= 1
        variants_in_batch = {p.get("variant") for p in (obs.direction_batch or [])}
        # The batch should sample from at least two of the four variants.
        assert len(variants_in_batch & {"pert_dir", "pert_de", "gse_pert", "gse_gene"}) >= 2

    def test_submission_scores(self):
        obs = self.env.reset(task_id="pertbench_000")
        answers = {p["pair_id"]: "Decrease" for p in (obs.direction_batch or [])}
        result = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            direction_answers=answers,
        ))
        assert result.done is True
        assert 0.01 <= result.reward <= 0.99
        breakdown = result.metadata.get("score_breakdown") if result.metadata else {}
        assert "per_variant" in breakdown

    def test_deterministic_replay(self):
        obs1 = self.env.reset(task_id="pertbench_003")
        obs2 = self.env.reset(task_id="pertbench_003")
        ids1 = [p["pair_id"] for p in (obs1.direction_batch or [])]
        ids2 = [p["pair_id"] for p in (obs2.direction_batch or [])]
        assert ids1 == ids2


# ── v3: get_structure tool inside labs ──────────────────────────────────

class TestGetStructureTool:
    def setup_method(self):
        self.env = BioresearchEnvironment()

    def test_get_structure_in_protein_lab(self):
        obs = self.env.reset(task_type="protein_hypothesis_lab")
        pid = obs.context.get("protein_id")
        assert pid
        # get_structure must be advertised in the lab tools.
        assert "get_structure" in (obs.available_tools or [])
        result = self.env.step(BioresearchAction(
            task_id=obs.task_id,
            tool_name="get_structure",
            tool_args={"protein_id": pid},
        ))
        # Regardless of whether this protein has a structure_path, the tool
        # must return a structured response and not crash the episode.
        assert result.tool_result is not None
        assert "source" in (result.tool_result or {}) or "error" in (result.tool_result or {})
