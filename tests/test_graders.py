"""Unit tests for bioresearch grading functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.graders import (
    grade_dna_classification,
    grade_dna_reasoning,
    grade_evidence_ranking,
    grade_protein_function,
    grade_leaf_go_f1,
    grade_process_trace,
    grade_intervention,
    grade_tool_efficiency,
)
from server.data_loader import ProteinSample


# ── Helpers ──────────────────────────────────────────────────────────────

def assert_clamped(score):
    assert 0.01 <= score <= 0.99, f"Score {score} not in [0.01, 0.99]"


# ── Task 1: DNA Classification ──────────────────────────────────────────

class TestDNAClassification:
    def test_exact_match(self):
        score, bd = grade_dna_classification("cushing syndrome", "cushing syndrome")
        assert_clamped(score)
        assert score >= 0.85
        assert bd["match_type"] == "exact"

    def test_case_insensitive(self):
        score, _ = grade_dna_classification("Cushing Syndrome", "cushing syndrome")
        assert score >= 0.85

    def test_partial_match(self):
        score, bd = grade_dna_classification("cushing", "cushing syndrome")
        assert_clamped(score)
        assert 0.20 <= score <= 0.85

    def test_no_match(self):
        score, bd = grade_dna_classification("diabetes", "cushing syndrome")
        assert_clamped(score)
        assert score <= 0.30

    def test_empty_input(self):
        score, _ = grade_dna_classification("", "cushing syndrome")
        assert_clamped(score)
        assert score == 0.01

    def test_long_input(self):
        score, _ = grade_dna_classification("a" * 10000, "cushing syndrome")
        assert_clamped(score)


# ── Task 2: DNA Reasoning ───────────────────────────────────────────────

class TestDNAReasoning:
    def test_correct_answer_good_reasoning(self):
        reasoning = (
            "Step 1: The mutation in PDE11A causes loss of function in the phosphodiesterase enzyme.\n"
            "Step 2: This leads to elevated cAMP levels in the cell.\n"
            "Step 3: Elevated cAMP activates PKA (protein kinase A) signaling pathway.\n"
            "Step 4: PKA phosphorylates CREB transcription factors.\n"
            "Step 5: This results in increased cortisol production, causing Cushing syndrome."
        )
        score, bd = grade_dna_reasoning(
            "cushing syndrome", reasoning,
            "cushing syndrome",
            "PDE11A loss of function leads to elevated cAMP which activates PKA and CREB resulting in cortisol overproduction causing Cushing syndrome",
            pathway_genes=["PDE11A", "PRKAR1A", "PRKACA", "CREB1", "STAR", "CYP11B1"],
        )
        assert_clamped(score)
        assert score >= 0.40

    def test_correct_answer_no_reasoning(self):
        score, bd = grade_dna_reasoning(
            "cushing syndrome", "",
            "cushing syndrome", "gold reasoning", pathway_genes=["PDE11A"],
        )
        assert_clamped(score)
        assert score <= 0.40

    def test_wrong_answer_good_reasoning(self):
        score, bd = grade_dna_reasoning(
            "diabetes", "Step 1: PDE11A mutation causes issues.",
            "cushing syndrome", "gold", pathway_genes=["PDE11A"],
        )
        assert_clamped(score)
        assert score < 0.50

    def test_empty_everything(self):
        score, _ = grade_dna_reasoning("", "", "cushing syndrome", "gold", [])
        assert_clamped(score)


# ── Task 3: Protein Function ────────────────────────────────────────────

SAMPLE_PROTEIN = ProteinSample(
    row_idx=0,
    protein_id="Q708S6",
    protein_names="Acid-sensing ion channel 1C",
    protein_function="Forms voltage-independent pH-gated trimeric sodium channels",
    organism="Danio rerio",
    length=529.0,
    subcellular_location="Cell membrane; Multi-pass membrane protein",
    sequence="MTAMKGDS...",
    go_ids=["GO:0005575", "GO:0005886", "GO:0016020"],
    go_bp=[],
    go_mf=[],
    go_cc=["GO:0005575", "GO:0005886", "GO:0016020"],
    interpro_ids=["IPR001873"],
    interpro_formatted="- IPR001873: Epithelial sodium channel (family) [20-461]",
    ppi_formatted="- Membrane-associated guanylate kinase",
    go_pred="",
)


class TestProteinFunction:
    def test_good_prediction(self):
        score, bd = grade_protein_function(
            predicted_function="pH-gated sodium channel forming trimeric complexes",
            predicted_location="Cell membrane",
            predicted_go_terms=["GO:0005886", "GO:0016020"],
            predicted_reasoning="The protein contains epithelial sodium channel domains",
            gold=SAMPLE_PROTEIN,
        )
        assert_clamped(score)
        assert score >= 0.20

    def test_wrong_prediction(self):
        score, _ = grade_protein_function(
            predicted_function="DNA repair enzyme",
            predicted_location="Nucleus",
            predicted_go_terms=["GO:9999999"],
            predicted_reasoning="This is a nuclear protein",
            gold=SAMPLE_PROTEIN,
        )
        assert_clamped(score)
        assert score <= 0.30

    def test_empty_prediction(self):
        score, _ = grade_protein_function("", None, None, None, SAMPLE_PROTEIN)
        assert_clamped(score)

    def test_go_terms_matching(self):
        score, bd = grade_protein_function(
            predicted_function="channel protein",
            predicted_location="membrane",
            predicted_go_terms=["GO:0005575", "GO:0005886", "GO:0016020"],
            predicted_reasoning="reasoning",
            gold=SAMPLE_PROTEIN,
        )
        assert bd["go_f1"] > 0.5


# ── Task 4: Evidence Ranking ────────────────────────────────────────────

class TestEvidenceRanking:
    def test_perfect_ranking(self):
        score, bd = grade_evidence_ranking(
            selected_disease="cushing syndrome",
            ranked_diseases=["cushing syndrome", "parkinsons disease", "als", "diabetes"],
            elimination_reasoning={
                "parkinsons disease": "This pathway involves cAMP and cortisol, not dopamine neurons. The genes are unrelated to SNCA.",
                "als": "ALS involves motor neuron degeneration which is a different mechanism entirely.",
                "diabetes": "The pathway leads to cortisol not insulin, so diabetes is not associated.",
            },
            supporting_evidence="Step 1: PDE11A mutation leads to loss of function. Step 2: cAMP accumulates. Step 3: PKA activates cortisol production.",
            gold_disease="cushing syndrome",
            gold_reasoning="PDE11A mutation causes cortisol overproduction",
            distractors=["parkinsons disease", "als", "diabetes"],
            pathway_genes=["PDE11A", "PRKAR1A", "PRKACA"],
        )
        assert_clamped(score)
        assert score >= 0.40

    def test_wrong_ranking(self):
        score, bd = grade_evidence_ranking(
            selected_disease="diabetes",
            ranked_diseases=["diabetes", "als", "parkinsons disease", "cushing syndrome"],
            elimination_reasoning={},
            supporting_evidence="",
            gold_disease="cushing syndrome",
            gold_reasoning="gold",
            distractors=["parkinsons disease", "als", "diabetes"],
        )
        assert_clamped(score)
        assert score <= 0.30

    def test_empty_inputs(self):
        score, _ = grade_evidence_ranking(
            selected_disease="", ranked_diseases=None, elimination_reasoning=None,
            supporting_evidence=None, gold_disease="cushing syndrome",
            gold_reasoning="gold", distractors=["a", "b", "c"],
        )
        assert_clamped(score)


# ── GRPO Variance Test ──────────────────────────────────────────────────

class TestGRPOVariance:
    """Verify that different quality responses produce sufficient score spread."""

    def test_classification_variance(self):
        scores = []
        for pred in ["cushing syndrome", "cushing", "parkinsons disease", "unknown", ""]:
            s, _ = grade_dna_classification(pred, "cushing syndrome")
            scores.append(s)
        spread = max(scores) - min(scores)
        assert spread >= 0.40, f"Score spread {spread} too low for GRPO"

    def test_reasoning_variance(self):
        gold = "PDE11A loss causes elevated cAMP activating PKA and CREB for cortisol Cushing syndrome"
        genes = ["PDE11A", "PRKAR1A", "PRKACA", "CREB1"]
        responses = [
            ("cushing syndrome", "Step 1: PDE11A mutation. Step 2: cAMP elevation. Step 3: PKA activation. Step 4: CREB. Step 5: Cortisol leads to Cushing syndrome."),
            ("cushing syndrome", "The mutation causes disease."),
            ("diabetes", "Something about insulin."),
            ("unknown", ""),
        ]
        scores = []
        for ans, reasoning in responses:
            s, _ = grade_dna_reasoning(ans, reasoning, "cushing syndrome", gold, genes)
            scores.append(s)
        spread = max(scores) - min(scores)
        assert spread >= 0.30, f"Score spread {spread} too low for GRPO"


# ── Lab graders (long-horizon tasks) ─────────────────────────────────────

class TestLeafGoF1:
    def test_exact_match(self):
        gold_text = "go_bp_leaf: GO:0006915 (apoptotic process); go_mf_leaf: GO:0003677 (DNA binding)"
        score, bd = grade_leaf_go_f1(["GO:0006915", "GO:0003677"], gold_text)
        assert_clamped(score)
        assert bd["f1"] == 1.0
        assert score >= 0.95

    def test_no_prediction(self):
        gold_text = "GO:0006915"
        score, bd = grade_leaf_go_f1([], gold_text)
        assert score <= 0.05
        assert bd["f1"] == 0.0

    def test_no_gold_neutral(self):
        score, bd = grade_leaf_go_f1(["GO:0006915"], "")
        assert 0.40 <= score <= 0.60

    def test_partial_match(self):
        gold_text = "GO:0006915, GO:0003677, GO:0005515"
        score, bd = grade_leaf_go_f1(["GO:0006915", "GO:9999999"], gold_text)
        assert 0.10 < score < 0.80
        assert 0.0 < bd["f1"] < 1.0

    def test_variance_for_grpo(self):
        gold_text = "GO:0006915, GO:0003677, GO:0005515"
        candidates = [
            ["GO:0006915", "GO:0003677", "GO:0005515"],   # perfect
            ["GO:0006915", "GO:0003677"],                 # 2/3
            ["GO:0006915"],                               # 1/3
            [],                                           # none
        ]
        scores = [grade_leaf_go_f1(c, gold_text)[0] for c in candidates]
        assert scores[0] > scores[1] > scores[2] > scores[3]


class TestProcessTrace:
    def test_identical_steps_score_high(self):
        steps = [
            "Step 1: Inspect the InterPro domains for conserved kinase folds.",
            "Step 2: Look up the PPI network to find TP53 co-regulators.",
            "Step 3: Check GO terms for DNA damage response signatures.",
        ]
        score, bd = grade_process_trace(steps, steps)
        assert_clamped(score)
        assert score >= 0.90

    def test_disjoint_steps_score_low(self):
        pred = ["Step 1: Random unrelated text about cats and dogs."]
        gold = ["Step 1: Look up the protein sequence and compute hydrophobicity profile."]
        score, _ = grade_process_trace(pred, gold)
        assert score <= 0.60

    def test_no_gold_neutral(self):
        score, bd = grade_process_trace(["something"], [])
        assert 0.40 <= score <= 0.60
        assert bd["gold_step_count"] == 0

    def test_variance(self):
        gold = [
            "Step 1: Look up the InterPro domains for the kinase family.",
            "Step 2: Check the PPI network for upstream regulators.",
        ]
        perfect = gold
        partial = [gold[0], "unrelated sentence"]
        wrong = ["totally off-topic", "nothing to do with proteins"]
        s1, _ = grade_process_trace(perfect, gold)
        s2, _ = grade_process_trace(partial, gold)
        s3, _ = grade_process_trace(wrong, gold)
        assert s1 > s2 > s3


def _make_protein_sample(**kwargs) -> ProteinSample:
    defaults = dict(
        row_idx=0,
        protein_id="P04637",
        protein_names="Tumor suppressor p53 TP53",
        protein_function="DNA binding transcription factor that activates apoptosis",
        organism="Homo sapiens",
        length=393.0,
        subcellular_location="Nucleus",
        sequence="MEEPQSDPSVEPPLSQETFSDLWKLLPEN",
        go_ids=["GO:0003677", "GO:0006915"],
        interpro_formatted="IPR002117 p53 DNA-binding domain",
    )
    defaults.update(kwargs)
    return ProteinSample(**defaults)


class TestIntervention:
    def test_none_returns_min(self):
        sample = _make_protein_sample()
        score, bd = grade_intervention(None, sample)
        assert score <= 0.05

    def test_missing_fields(self):
        sample = _make_protein_sample()
        score, _ = grade_intervention({"mode": "inhibit"}, sample)
        assert score <= 0.10

    def test_plausible_target_and_mode(self):
        sample = _make_protein_sample()
        score, bd = grade_intervention(
            {"mode": "activate", "target": "TP53"},
            sample,
        )
        assert_clamped(score)
        assert score >= 0.45
        assert bd["mode"] == "activate"

    def test_invalid_mode_scored_low(self):
        sample = _make_protein_sample()
        score, _ = grade_intervention(
            {"mode": "banana", "target": "TP53"},
            sample,
        )
        assert score <= 0.60


class TestToolEfficiency:
    def test_no_calls_neutral(self):
        score, bd = grade_tool_efficiency([], "some reasoning")
        assert 0.40 <= score <= 0.60
        assert bd.get("note") == "no tool calls"

    def test_useful_calls_score_high(self):
        calls = [
            {"tool_name": "get_interpro", "tool_args": {"gene": "TP53"}, "result": {"domain": "p53_DNA_binding_domain"}},
            {"tool_name": "get_ppi", "tool_args": {"gene": "TP53"}, "result": {"partners": "MDM2 ATM CHEK2"}},
        ]
        reasoning = "The p53_DNA_binding_domain is central, and MDM2 is a key negative regulator."
        score, bd = grade_tool_efficiency(calls, reasoning)
        assert_clamped(score)
        assert score >= 0.40

    def test_redundant_calls_penalised(self):
        call = {"tool_name": "get_interpro", "tool_args": {"gene": "TP53"}, "result": {"domain": "x"}}
        useful_score, _ = grade_tool_efficiency([call], "x domain is informative")
        redundant_score, _ = grade_tool_efficiency([call, call, call, call], "x domain is informative")
        assert redundant_score < useful_score
