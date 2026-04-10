"""Unit tests for bioresearch grading functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.graders import (
    grade_dna_classification,
    grade_dna_reasoning,
    grade_evidence_ranking,
    grade_protein_function,
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
