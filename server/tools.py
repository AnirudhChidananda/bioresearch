"""
Tool implementations for the Virtual Tumor Board multi-agent environment.

Every tool is a DETERMINISTIC pure function of (task_id, args).
This is a hard requirement for GRPO same-prompt replay: the identical
tool call at the identical turn must always return the identical output.

Tools exposed to the orchestrator agent:
    - blast_lookup(sequence)         → nearest-known-gene summary
    - pathway_expand(gene)           → pathway neighbours
    - go_term_lookup(protein_or_gene)→ GO annotations
    - literature_snippet(disease)    → canned abstract
    - ask_specialist(role, question) → specialist actor response
    - submit_consensus(answer, reasoning) → terminates episode

The last tool (submit_consensus) is handled by the environment itself,
not this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .data_loader import DNASample, ProteinSample
except ImportError:  # pragma: no cover
    from server.data_loader import DNASample, ProteinSample  # type: ignore


TOOL_NAMES = {
    "blast_lookup",
    "pathway_expand",
    "go_term_lookup",
    "literature_snippet",
    "ask_specialist",
    "submit_consensus",
}


_LITERATURE_CACHE: Optional[Dict[str, Any]] = None


def _load_literature() -> Dict[str, Any]:
    global _LITERATURE_CACHE
    if _LITERATURE_CACHE is not None:
        return _LITERATURE_CACHE

    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "literature.json",
        Path("/app/env/data/literature.json"),
    ]
    for p in candidates:
        if p.is_file():
            with open(p, "r") as f:
                _LITERATURE_CACHE = json.load(f)
            return _LITERATURE_CACHE

    _LITERATURE_CACHE = {}
    return _LITERATURE_CACHE


def _extract_pathway_genes(question: str) -> List[str]:
    """Extract gene symbols from the pathway gene list in a DNA question."""
    genes: List[str] = []
    section = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    if section:
        for match in re.finditer(r"(\w+)\s*;", section.group(1)):
            genes.append(match.group(1))
    return genes


# ─── Public tool implementations ─────────────────────────────────────────

def blast_lookup(sample, sequence_snippet: str = "") -> str:
    """
    Pretend-BLAST: returns the nearest-known-gene context for the sample.

    For DNA samples: returns pathway gene list + mutation description.
    For protein samples: returns protein name + organism + nearest domain.
    """
    if isinstance(sample, DNASample):
        genes = _extract_pathway_genes(sample.question)
        genes_preview = ", ".join(genes[:6]) if genes else "unknown"
        chromosome = "unknown"
        chrom_match = re.search(r"chromosome\s+(\S+)", sample.question, re.IGNORECASE)
        if chrom_match:
            chromosome = chrom_match.group(1)
        return (
            f"BLAST hit: variant on chromosome {chromosome}. "
            f"Pathway gene cluster ({len(genes)} genes): {genes_preview}. "
            f"Variant length: {len(sample.variant_sequence)} bp."
        )
    if isinstance(sample, ProteinSample):
        domain = sample.interpro_formatted.split(";")[0].strip() if sample.interpro_formatted else "no domain hit"
        return (
            f"BLAST hit: {sample.protein_names} ({sample.organism}). "
            f"Top InterPro domain: {domain}. Length: {int(sample.length)} aa."
        )
    return "BLAST: no hit."


def pathway_expand(sample, gene: str) -> str:
    """
    Expand a gene into its pathway neighbours (from the sample's own pathway list).
    """
    gene_upper = gene.strip().upper()
    if not gene_upper:
        return "pathway_expand: provide a gene symbol."

    if isinstance(sample, DNASample):
        genes = _extract_pathway_genes(sample.question)
        gene_set_upper = {g.upper() for g in genes}
        if gene_upper in gene_set_upper:
            neighbours = [g for g in genes if g.upper() != gene_upper][:8]
            return (
                f"{gene_upper} is part of the case pathway. "
                f"Direct neighbours: {', '.join(neighbours) if neighbours else 'none listed'}."
            )
        return f"{gene_upper} is NOT in the case pathway. Closest listed genes: {', '.join(genes[:5]) if genes else 'none'}."

    if isinstance(sample, ProteinSample):
        if gene_upper in sample.protein_names.upper() or gene_upper == sample.protein_id.upper():
            ppi_preview = sample.ppi_formatted.split(";")[:5] if sample.ppi_formatted else []
            return (
                f"{gene_upper}: known interactors — {', '.join(ppi_preview) if ppi_preview else 'none annotated'}."
            )
        return f"{gene_upper} not directly linked to this protein case."
    return "pathway_expand: no sample context available."


def go_term_lookup(sample, query: str = "") -> str:
    """
    Return GO annotations. For protein samples, returns real GO terms.
    For DNA samples, returns GO terms implied by pathway genes (synthetic but deterministic).
    """
    if isinstance(sample, ProteinSample):
        preview = sample.go_ids[:8]
        if not preview:
            return f"go_term_lookup: no GO annotations available for {sample.protein_names}."
        return (
            f"GO annotations for {sample.protein_names}: {', '.join(preview)}"
            + (f" ... and {len(sample.go_ids) - 8} more." if len(sample.go_ids) > 8 else "")
        )
    if isinstance(sample, DNASample):
        genes = _extract_pathway_genes(sample.question)
        return (
            f"GO lookup for pathway genes: functional category inferred from "
            f"{len(genes)} gene symbols in the pathway neighborhood. "
            f"Key genes for GO annotation: {', '.join(genes[:4]) if genes else 'none'}."
        )
    return "go_term_lookup: no sample context."


def literature_snippet(sample, disease: str) -> str:
    """
    Return a canned literature abstract for the queried disease.
    """
    lit = _load_literature()
    disease_key = disease.strip().lower()

    for key, snippet in lit.items():
        if key.lower() == disease_key or key.lower() in disease_key or disease_key in key.lower():
            return f"[lit] {snippet}"

    return (
        f"[lit] No curated abstract indexed for '{disease}'. "
        f"Available topics: {', '.join(list(lit.keys())[:5])}..."
    )


def ask_specialist(sample, role: str, question: str, actors_module) -> str:
    """
    Delegate a question to a specialist actor. Returns the specialist's answer.
    Deterministic from (task_id, role, normalized_question).
    """
    return actors_module.respond(sample, role, question)


# ─── Validation / dispatch helpers ───────────────────────────────────────

def validate_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
    """Return an error message if the call is invalid, else None."""
    if tool_name not in TOOL_NAMES:
        return f"Unknown tool: {tool_name}. Valid: {sorted(TOOL_NAMES)}"
    if not isinstance(tool_args, dict):
        return "tool_args must be a dict."

    required_args = {
        "blast_lookup": [],
        "pathway_expand": ["gene"],
        "go_term_lookup": [],
        "literature_snippet": ["disease"],
        "ask_specialist": ["role", "question"],
        "submit_consensus": ["answer"],
    }
    for req in required_args.get(tool_name, []):
        if req not in tool_args or tool_args[req] is None:
            return f"Tool {tool_name} requires argument '{req}'."
    return None


def dispatch(sample, tool_name: str, tool_args: Dict[str, Any], actors_module) -> str:
    """Dispatch a tool call to its implementation. Assumes already validated."""
    if tool_name == "blast_lookup":
        return blast_lookup(sample, tool_args.get("sequence", ""))
    if tool_name == "pathway_expand":
        return pathway_expand(sample, tool_args.get("gene", ""))
    if tool_name == "go_term_lookup":
        return go_term_lookup(sample, tool_args.get("query", ""))
    if tool_name == "literature_snippet":
        return literature_snippet(sample, tool_args.get("disease", ""))
    if tool_name == "ask_specialist":
        return ask_specialist(
            sample,
            role=tool_args.get("role", ""),
            question=tool_args.get("question", ""),
            actors_module=actors_module,
        )
    return "Unsupported tool."
