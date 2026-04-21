"""Tests for the tool implementations (server/tools.py).

Tools MUST be deterministic: (task_id, args) → same output every time.
This is a hard requirement for GRPO same-prompt replay.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import actors as actors_module
from server import tools as tools_module
from server.data_loader import DataLoader


class TestToolValidation:
    def test_unknown_tool_is_rejected(self):
        err = tools_module.validate_tool_call("nonexistent_tool", {})
        assert err is not None
        assert "Unknown tool" in err

    def test_missing_required_arg_is_rejected(self):
        err = tools_module.validate_tool_call("pathway_expand", {})
        assert err is not None
        assert "gene" in err

    def test_valid_call_has_no_error(self):
        assert tools_module.validate_tool_call("pathway_expand", {"gene": "PDE11A"}) is None

    def test_non_dict_args_are_rejected(self):
        err = tools_module.validate_tool_call("blast_lookup", "not a dict")
        assert err is not None


class TestToolDeterminism:
    """Every tool must return the identical output when called twice with identical args."""

    def setup_method(self):
        self.data = DataLoader()

    def test_blast_lookup_deterministic(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        a = tools_module.blast_lookup(sample)
        b = tools_module.blast_lookup(sample)
        assert a == b
        assert "BLAST" in a

    def test_pathway_expand_deterministic(self):
        sample = self.data.get_dna_sample_by_id("dna_007")
        a = tools_module.pathway_expand(sample, "PDE11A")
        b = tools_module.pathway_expand(sample, "PDE11A")
        assert a == b

    def test_go_term_lookup_deterministic_protein(self):
        sample = self.data.get_protein_sample_by_id("protein_000")
        a = tools_module.go_term_lookup(sample)
        b = tools_module.go_term_lookup(sample)
        assert a == b

    def test_literature_snippet_known_disease(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        a = tools_module.literature_snippet(sample, "cushing syndrome")
        b = tools_module.literature_snippet(sample, "cushing syndrome")
        assert a == b
        assert "[lit]" in a

    def test_literature_snippet_unknown_disease(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        out = tools_module.literature_snippet(sample, "imaginary disease xyz")
        assert "[lit]" in out

    def test_ask_specialist_deterministic(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        a = tools_module.ask_specialist(sample, "geneticist", "variant impact?", actors_module)
        b = tools_module.ask_specialist(sample, "geneticist", "variant impact?", actors_module)
        assert a == b


class TestToolDispatch:
    def setup_method(self):
        self.data = DataLoader()

    def test_dispatch_blast(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        out = tools_module.dispatch(sample, "blast_lookup", {}, actors_module)
        assert "BLAST" in out

    def test_dispatch_ask_specialist(self):
        sample = self.data.get_dna_sample_by_id("dna_000")
        out = tools_module.dispatch(
            sample,
            "ask_specialist",
            {"role": "geneticist", "question": "x"},
            actors_module,
        )
        assert "[geneticist]" in out
