"""
Specialist actors for the Virtual Tumor Board environment.

Each specialist is a DETERMINISTIC responder: (sample, role, question) →
textual answer. Specialists never hallucinate — their responses are
grounded in the curated dataset fields (reasoning chains, pathway gene
lists, InterPro domains, etc.). This preserves GRPO replay semantics
while giving the orchestrator agent useful, domain-authentic context.

The design intentionally keeps actors cheap (no extra LLM calls)
so that GRPO training remains fast and reproducible.
"""

from __future__ import annotations

import re
from typing import List, Optional

try:
    from .data_loader import DNASample, ProteinSample
except ImportError:  # pragma: no cover
    from server.data_loader import DNASample, ProteinSample  # type: ignore


ROLES = ("geneticist", "pathway_analyst", "structural_biologist", "clinician")


ROLE_DESCRIPTIONS = {
    "geneticist": "Variant interpretation specialist. Asks about variant type, zygosity, gene-disease association.",
    "pathway_analyst": "Systems-biology specialist. Maps variants to pathway consequences.",
    "structural_biologist": "Protein-structure specialist. Reasons about domains, folds, catalytic residues.",
    "clinician": "Rare-disease clinician. Connects molecular findings to clinical phenotype.",
}


def list_roles() -> List[str]:
    return list(ROLES)


def normalise_role(role: str) -> str:
    r = (role or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "genetics": "geneticist",
        "variant": "geneticist",
        "pathway": "pathway_analyst",
        "systems_biology": "pathway_analyst",
        "systems_biologist": "pathway_analyst",
        "structure": "structural_biologist",
        "structural": "structural_biologist",
        "biologist": "structural_biologist",
        "doctor": "clinician",
        "physician": "clinician",
        "clinical": "clinician",
    }
    return aliases.get(r, r)


def _extract_pathway_genes(question: str) -> List[str]:
    genes: List[str] = []
    section = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    if section:
        for match in re.finditer(r"(\w+)\s*;", section.group(1)):
            genes.append(match.group(1))
    return genes


def _first_sentence(text: str, max_chars: int = 280) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=2)
    summary = parts[0] if parts else text
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary


# ─── Per-role response builders ──────────────────────────────────────────

def _geneticist_response(sample, question: str) -> str:
    if isinstance(sample, DNASample):
        ref_len = len(sample.reference_sequence)
        var_len = len(sample.variant_sequence)
        delta = var_len - ref_len
        variant_type = "indel" if delta != 0 else "substitution"
        ref_preview = sample.reference_sequence[:15]
        var_preview = sample.variant_sequence[:15]
        first_step = _first_sentence(sample.reasoning)
        return (
            f"[geneticist] Variant type: {variant_type} "
            f"(ref={ref_len} bp, var={var_len} bp, Δ={delta:+d}). "
            f"Reference 5'-start: {ref_preview}... Variant 5'-start: {var_preview}... "
            f"Key mechanism: {first_step}"
        )
    if isinstance(sample, ProteinSample):
        return (
            f"[geneticist] {sample.protein_names} ({sample.organism}): "
            f"{int(sample.length)} aa. No DNA-level variant info in a protein-only case; "
            f"defer to structural_biologist and pathway_analyst."
        )
    return "[geneticist] Insufficient case material to opine."


def _pathway_analyst_response(sample, question: str) -> str:
    if isinstance(sample, DNASample):
        genes = _extract_pathway_genes(sample.question)
        if not genes:
            return "[pathway_analyst] No explicit pathway gene list in this case."
        gene_preview = ", ".join(genes[:8])
        reasoning_first = _first_sentence(sample.reasoning)
        return (
            f"[pathway_analyst] Case pathway contains {len(genes)} genes "
            f"({gene_preview}{'...' if len(genes) > 8 else ''}). "
            f"Pathway-level consequence: {reasoning_first}"
        )
    if isinstance(sample, ProteinSample):
        ppi_preview = sample.ppi_formatted[:240] if sample.ppi_formatted else ""
        if not ppi_preview:
            return f"[pathway_analyst] {sample.protein_names}: no curated PPI data available."
        return (
            f"[pathway_analyst] {sample.protein_names} interactors / pathway context: "
            f"{ppi_preview}{'...' if len(sample.ppi_formatted) > 240 else ''}"
        )
    return "[pathway_analyst] No pathway context."


def _structural_biologist_response(sample, question: str) -> str:
    if isinstance(sample, ProteinSample):
        interpro = sample.interpro_formatted[:260] if sample.interpro_formatted else ""
        if not interpro:
            return (
                f"[structural_biologist] {sample.protein_names}: "
                f"{int(sample.length)} aa, no InterPro domain signature available."
            )
        loc_hint = f" Subcellular: {sample.subcellular_location}." if sample.subcellular_location else ""
        return (
            f"[structural_biologist] {sample.protein_names} domain architecture: "
            f"{interpro}{'...' if len(sample.interpro_formatted) > 260 else ''}.{loc_hint}"
        )
    if isinstance(sample, DNASample):
        return (
            f"[structural_biologist] This is a DNA-level case. "
            f"Variant region length: {len(sample.variant_sequence)} bp — "
            f"pathway-level effects dominate over structural concerns."
        )
    return "[structural_biologist] No structural context."


def _clinician_response(sample, question: str) -> str:
    if isinstance(sample, DNASample):
        disease_hint = sample.answer.strip()
        reasoning_first = _first_sentence(sample.reasoning, max_chars=200)
        return (
            f"[clinician] Clinical summary: phenotype consistent with {disease_hint}. "
            f"Molecular-to-clinical bridge: {reasoning_first}"
        )
    if isinstance(sample, ProteinSample):
        loc = sample.subcellular_location or "unspecified localisation"
        func_first = _first_sentence(sample.protein_function, max_chars=180)
        return (
            f"[clinician] {sample.protein_names}: {loc}. "
            f"Functional summary: {func_first}"
        )
    return "[clinician] No clinical context."


_ROLE_HANDLERS = {
    "geneticist": _geneticist_response,
    "pathway_analyst": _pathway_analyst_response,
    "structural_biologist": _structural_biologist_response,
    "clinician": _clinician_response,
}


def respond(sample, role: str, question: str) -> str:
    """Return the specialist's textual response to a question about this case."""
    role_norm = normalise_role(role)
    handler = _ROLE_HANDLERS.get(role_norm)
    if handler is None:
        return (
            f"[specialist] Unknown role '{role}'. "
            f"Valid roles: {', '.join(ROLES)}."
        )
    return handler(sample, question or "")


def relevant_specialists(sample) -> List[str]:
    """
    Return the ordered list of specialists most useful for this case type.
    Used by the consensus grader to reward good delegation.
    """
    if isinstance(sample, DNASample):
        return ["geneticist", "pathway_analyst", "clinician"]
    if isinstance(sample, ProteinSample):
        return ["structural_biologist", "pathway_analyst", "clinician"]
    return list(ROLES)
