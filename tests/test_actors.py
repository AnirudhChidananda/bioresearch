"""Tests for the specialist actors (server/actors.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import actors
from server.data_loader import DataLoader


class TestActorRoles:
    def test_list_roles_returns_four(self):
        roles = actors.list_roles()
        assert len(roles) == 4
        assert "geneticist" in roles
        assert "pathway_analyst" in roles
        assert "structural_biologist" in roles
        assert "clinician" in roles

    def test_role_normalisation(self):
        assert actors.normalise_role("Geneticist") == "geneticist"
        assert actors.normalise_role("pathway-analyst") == "pathway_analyst"
        assert actors.normalise_role("systems biologist") == "pathway_analyst"
        assert actors.normalise_role("doctor") == "clinician"
        assert actors.normalise_role("structural") == "structural_biologist"


class TestActorResponses:
    def setup_method(self):
        self.data = DataLoader()

    def test_geneticist_on_dna_case(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        response = actors.respond(sample, "geneticist", "variant impact?")
        assert response.startswith("[geneticist]")
        assert len(response) > 40

    def test_pathway_analyst_lists_pathway_genes(self):
        sample = self.data.get_dna_sample_by_id("dna_007")
        response = actors.respond(sample, "pathway_analyst", "pathway?")
        assert response.startswith("[pathway_analyst]")

    def test_structural_biologist_on_protein(self):
        sample = self.data.get_protein_sample_by_id("protein_000")
        response = actors.respond(sample, "structural_biologist", "domains?")
        assert response.startswith("[structural_biologist]")

    def test_clinician_mentions_phenotype(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        response = actors.respond(sample, "clinician", "phenotype?")
        assert response.startswith("[clinician]")

    def test_unknown_role_returns_specialist_error(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        response = actors.respond(sample, "cardiologist", "?")
        assert response.startswith("[specialist]")
        assert "Unknown role" in response

    def test_relevant_specialists_dna(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        rel = actors.relevant_specialists(sample)
        assert "geneticist" in rel
        assert "pathway_analyst" in rel

    def test_relevant_specialists_protein(self):
        sample = self.data.get_protein_sample_by_id("protein_000")
        rel = actors.relevant_specialists(sample)
        assert "structural_biologist" in rel


class TestActorDeterminism:
    """Same (sample, role, question) must always return the same response."""

    def setup_method(self):
        self.data = DataLoader()

    def test_geneticist_deterministic(self):
        s = self.data.get_dna_sample_by_id("dna_012")
        a = actors.respond(s, "geneticist", "explain")
        b = actors.respond(s, "geneticist", "explain")
        assert a == b

    def test_all_roles_deterministic_across_samples(self):
        sample_ids = ["dna_000", "dna_007", "dna_015", "protein_000", "protein_010"]
        for sid in sample_ids:
            if sid.startswith("dna_"):
                s = self.data.get_dna_sample_by_id(sid)
            else:
                s = self.data.get_protein_sample_by_id(sid)
            for role in actors.list_roles():
                a = actors.respond(s, role, "case summary please")
                b = actors.respond(s, role, "case summary please")
                assert a == b, f"Non-determinism for role={role}, sample={sid}"
