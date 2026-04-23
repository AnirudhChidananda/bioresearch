"""
Grading functions for the Bioresearch Environment.

Each grader returns (score, breakdown_dict) where score is in [0.01, 0.99].
Grading is deterministic: same inputs always produce the same score.
Designed for GRPO compatibility with smooth, continuous reward signals
and step-level reasoning decomposition.
"""

import difflib
import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .data_loader import DiagnosisSample, DrugRecord, KeggSample, LigandSample, ProteinSample
except ImportError:
    from server.data_loader import DiagnosisSample, DrugRecord, KeggSample, LigandSample, ProteinSample


# =========================================================================
# Shared helpers for the new lab graders
# =========================================================================


_GO_ID_RE = re.compile(r"GO:\d{7}", re.IGNORECASE)


def _extract_go_ids(text: str) -> Set[str]:
    """Extract GO:XXXXXXX IDs from a free-text string, normalised uppercase."""
    if not text:
        return set()
    return {m.group(0).upper() for m in _GO_ID_RE.finditer(text)}


def _normalise_go_list(items: Optional[List[str]]) -> Set[str]:
    if not items:
        return set()
    out: Set[str] = set()
    for item in items:
        if not item:
            continue
        found = _extract_go_ids(item)
        if found:
            out |= found
        else:
            stripped = item.strip()
            if stripped.upper().startswith("GO:"):
                out.add(stripped.upper())
    return out


def _clamp(score: float) -> float:
    return max(0.01, min(0.99, score))


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenise(text: str) -> Set[str]:
    return set(_normalise(text).split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_steps(reasoning: str) -> List[str]:
    """Split reasoning into individual steps."""
    step_pattern = re.compile(r"(?:Step\s*\d+\s*[:.]|^\d+\s*[:.)]|\n\s*-\s+)", re.MULTILINE | re.IGNORECASE)
    parts = step_pattern.split(reasoning)
    steps = [p.strip() for p in parts if p and p.strip()]
    if len(steps) <= 1:
        sentences = re.split(r'(?<=[.!?])\s+', reasoning.strip())
        steps = [s.strip() for s in sentences if len(s.strip()) > 20]
    return steps


def _extract_gene_names(text: str) -> Set[str]:
    """Extract plausible gene/protein names (uppercase tokens of 2-10 chars or known patterns)."""
    gene_pattern = re.compile(r'\b[A-Z][A-Z0-9]{1,9}\b')
    return set(gene_pattern.findall(text))


def _extract_pathway_genes(question: str) -> List[str]:
    """Extract gene names from the pathway context in the question."""
    genes = []
    gene_section = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    if gene_section:
        for match in re.finditer(r"(\w+)\s*;", gene_section.group(1)):
            genes.append(match.group(1))
    return genes


CAUSAL_CONNECTORS = {
    "leads to", "results in", "causes", "activates", "inhibits",
    "promotes", "induces", "reduces", "increases", "decreases",
    "impairs", "disrupts", "enhances", "stimulates", "blocks",
    "downregulates", "upregulates", "phosphorylates", "triggers",
}


def _count_causal_connectors(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for c in CAUSAL_CONNECTORS if c in text_lower)


# =========================================================================
# Task 1: DNA Mutation Disease Classification
# =========================================================================

def grade_dna_classification(predicted: str, gold: str) -> Tuple[float, Dict[str, Any]]:
    pred_norm = _normalise(predicted)
    gold_norm = _normalise(gold)

    pred_tokens = _tokenise(predicted)
    gold_tokens = _tokenise(gold)

    if pred_norm == gold_norm:
        score = 0.90
        match_type = "exact"
    else:
        jaccard = _jaccard(pred_tokens, gold_tokens)
        if jaccard > 0.5:
            score = 0.50 + (jaccard * 0.40)
            match_type = "jaccard_high"
        elif gold_tokens & pred_tokens:
            overlap_ratio = len(gold_tokens & pred_tokens) / len(gold_tokens)
            score = 0.20 + (overlap_ratio * 0.20)
            match_type = "partial"
        else:
            score = 0.01
            match_type = "no_match"

    breakdown = {
        "predicted": pred_norm,
        "gold": gold_norm,
        "match_type": match_type,
        "jaccard": _jaccard(pred_tokens, gold_tokens),
        "score": _clamp(score),
    }
    return _clamp(score), breakdown


# =========================================================================
# Task 2: DNA Mutation Biological Reasoning
# =========================================================================

def grade_dna_reasoning(
    predicted_answer: str,
    predicted_reasoning: str,
    gold_answer: str,
    gold_reasoning: str,
    pathway_genes: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    answer_score, answer_breakdown = grade_dna_classification(predicted_answer, gold_answer)
    answer_component = answer_score * 0.40

    if not predicted_reasoning or not predicted_reasoning.strip():
        return _clamp(answer_component), {
            "answer": answer_breakdown,
            "reasoning": {"error": "no reasoning provided"},
            "answer_component": answer_component,
            "reasoning_component": 0.0,
        }

    # Step count and structure (max 0.10)
    steps = _extract_steps(predicted_reasoning)
    step_count = len(steps)
    if step_count >= 5:
        step_structure_score = 0.10
    elif step_count >= 3:
        step_structure_score = 0.07
    elif step_count >= 1:
        step_structure_score = 0.03
    else:
        step_structure_score = 0.0

    # Biological concept coverage (max 0.20)
    gold_concepts = _extract_gene_names(gold_reasoning)
    gold_bio_terms = set()
    for term in re.findall(r'\b[a-z]{4,}\b', gold_reasoning.lower()):
        if term in {
            "mutation", "protein", "enzyme", "kinase", "receptor",
            "phosphorylation", "transcription", "degradation", "aggregation",
            "apoptosis", "mitochondrial", "ubiquitin", "autophagy",
            "oxidative", "neurodegeneration", "degeneration", "dysfunction",
            "pathway", "signaling", "cascade", "expression", "synthesis",
            "hormone", "cortisol", "dopamine", "insulin", "cholesterol",
        }:
            gold_bio_terms.add(term)

    all_gold_concepts = gold_concepts | gold_bio_terms
    if all_gold_concepts:
        pred_upper = _extract_gene_names(predicted_reasoning)
        pred_bio = set()
        for term in re.findall(r'\b[a-z]{4,}\b', predicted_reasoning.lower()):
            if term in gold_bio_terms:
                pred_bio.add(term)
        found = (pred_upper & gold_concepts) | pred_bio
        concept_coverage = len(found) / len(all_gold_concepts)
    else:
        concept_coverage = 0.0
    concept_score = min(concept_coverage, 1.0) * 0.20

    # Pathway gene coverage (max 0.15)
    if pathway_genes:
        pathway_set = {g.upper() for g in pathway_genes}
        pred_genes = _extract_gene_names(predicted_reasoning)
        pathway_coverage = len(pred_genes & pathway_set) / len(pathway_set) if pathway_set else 0.0
    else:
        pathway_coverage = 0.0
    pathway_score = min(pathway_coverage, 1.0) * 0.15

    # Causal chain coherence (max 0.10)
    causal_count = _count_causal_connectors(predicted_reasoning)
    causal_coherence = min(causal_count / 5.0, 1.0)
    causal_score = causal_coherence * 0.10

    # Hallucination penalty (max -0.05)
    hallucination_penalty = 0.0
    if pathway_genes:
        pathway_set_upper = {g.upper() for g in pathway_genes}
        pred_genes = _extract_gene_names(predicted_reasoning)
        hallucinated = pred_genes - pathway_set_upper - gold_concepts
        common_non_gene = {"DNA", "RNA", "ATP", "ADP", "AMP", "GTP", "GDP",
                           "NAD", "NADH", "FAD", "CAMP", "PKA", "PKC", "GDP",
                           "JSON", "STEP", "THE", "AND", "FOR", "NOT", "ARE",
                           "THIS", "THAT", "WITH", "FROM", "WILL", "HAS",
                           "HAVE", "BEEN", "EACH", "DOES", "ALSO", "INTO"}
        hallucinated = hallucinated - common_non_gene
        if len(hallucinated) > 3:
            hallucination_penalty = 0.05
        elif len(hallucinated) > 1:
            hallucination_penalty = 0.02

    reasoning_component = (
        step_structure_score + concept_score + pathway_score + causal_score - hallucination_penalty
    )
    reasoning_component = max(reasoning_component, 0.0)

    total = answer_component + reasoning_component

    breakdown = {
        "answer": answer_breakdown,
        "answer_component": round(answer_component, 4),
        "reasoning_component": round(reasoning_component, 4),
        "step_count": step_count,
        "step_structure_score": round(step_structure_score, 4),
        "concept_coverage": round(concept_coverage, 4),
        "concept_score": round(concept_score, 4),
        "pathway_coverage": round(pathway_coverage, 4),
        "pathway_score": round(pathway_score, 4),
        "causal_coherence": round(causal_coherence, 4),
        "causal_score": round(causal_score, 4),
        "hallucination_penalty": round(hallucination_penalty, 4),
    }
    return _clamp(total), breakdown


# =========================================================================
# Task 3: Protein Function Hypothesis Generation
# =========================================================================

def grade_protein_function(
    predicted_function: str,
    predicted_location: Optional[str],
    predicted_go_terms: Optional[List[str]],
    predicted_reasoning: Optional[str],
    gold: ProteinSample,
) -> Tuple[float, Dict[str, Any]]:
    # Function description (max 0.25)
    if predicted_function and predicted_function.strip():
        pred_tokens = _tokenise(predicted_function)
        gold_tokens = _tokenise(gold.protein_function)
        func_jaccard = _jaccard(pred_tokens, gold_tokens)
        func_score = func_jaccard * 0.25
    else:
        func_jaccard = 0.0
        func_score = 0.0

    # Subcellular location (max 0.20)
    if predicted_location and predicted_location.strip() and gold.subcellular_location:
        pred_loc_tokens = _tokenise(predicted_location)
        gold_loc_tokens = _tokenise(gold.subcellular_location)
        loc_jaccard = _jaccard(pred_loc_tokens, gold_loc_tokens)

        gold_loc_lower = gold.subcellular_location.lower()
        pred_loc_lower = predicted_location.lower()
        hierarchical_bonus = 0.0
        location_keywords = ["membrane", "cytoplasm", "nucleus", "mitochondri",
                             "endoplasmic", "golgi", "lysosom", "extracellular",
                             "secreted", "cell surface", "dendrite", "axon"]
        for kw in location_keywords:
            if kw in gold_loc_lower and kw in pred_loc_lower:
                hierarchical_bonus = max(hierarchical_bonus, 0.05)

        loc_score = min(loc_jaccard * 0.20 + hierarchical_bonus, 0.20)
    else:
        loc_jaccard = 0.0
        loc_score = 0.0

    # GO term prediction (max 0.35)
    gold_go_all = set(gold.go_ids + gold.go_bp + gold.go_mf + gold.go_cc)
    if predicted_go_terms and gold_go_all:
        pred_go_set = set()
        for term in predicted_go_terms:
            term = term.strip()
            if term.startswith("GO:"):
                pred_go_set.add(term)
            else:
                pred_go_set.add(term.lower())

        gold_go_normalised = set()
        for g in gold_go_all:
            g = g.strip()
            if g.startswith("GO:"):
                gold_go_normalised.add(g)
            else:
                gold_go_normalised.add(g.lower())

        true_positives = len(pred_go_set & gold_go_normalised)
        precision = true_positives / len(pred_go_set) if pred_go_set else 0.0
        recall = true_positives / len(gold_go_normalised) if gold_go_normalised else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        go_score = f1 * 0.35
    else:
        precision = recall = f1 = 0.0
        go_score = 0.0 if gold_go_all else 0.10

    # Reasoning quality (max 0.20)
    reasoning_score = 0.0
    reasoning_details: Dict[str, Any] = {}
    if predicted_reasoning and predicted_reasoning.strip():
        gold_func_concepts = _tokenise(gold.protein_function)
        pred_reasoning_tokens = _tokenise(predicted_reasoning)
        func_coverage = _jaccard(gold_func_concepts, pred_reasoning_tokens)
        reasoning_score += func_coverage * 0.08

        if gold.interpro_formatted:
            interpro_tokens = _tokenise(gold.interpro_formatted)
            interpro_overlap = len(pred_reasoning_tokens & interpro_tokens) / max(len(interpro_tokens), 1)
            reasoning_score += min(interpro_overlap, 1.0) * 0.06
            reasoning_details["interpro_overlap"] = round(interpro_overlap, 4)

        if gold.ppi_formatted:
            ppi_tokens = _tokenise(gold.ppi_formatted)
            ppi_overlap = len(pred_reasoning_tokens & ppi_tokens) / max(len(ppi_tokens), 1)
            reasoning_score += min(ppi_overlap, 1.0) * 0.06
            reasoning_details["ppi_overlap"] = round(ppi_overlap, 4)
        else:
            reasoning_score += 0.03

        reasoning_details["func_coverage"] = round(func_coverage, 4)
    reasoning_score = min(reasoning_score, 0.20)

    total = func_score + loc_score + go_score + reasoning_score

    breakdown = {
        "function_jaccard": round(func_jaccard, 4),
        "function_score": round(func_score, 4),
        "location_jaccard": round(loc_jaccard, 4),
        "location_score": round(loc_score, 4),
        "go_precision": round(precision, 4),
        "go_recall": round(recall, 4),
        "go_f1": round(f1, 4),
        "go_score": round(go_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "reasoning_details": reasoning_details,
    }
    return _clamp(total), breakdown


# =========================================================================
# Task 4: Variant Pathogenicity Evidence Ranking
# =========================================================================

def grade_evidence_ranking(
    selected_disease: str,
    ranked_diseases: Optional[List[str]],
    elimination_reasoning: Optional[Dict[str, str]],
    supporting_evidence: Optional[str],
    gold_disease: str,
    gold_reasoning: str,
    distractors: List[str],
    pathway_genes: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    gold_norm = _normalise(gold_disease)

    # Ranking accuracy (max 0.30)
    ranking_score = 0.0
    rank_position = -1
    if ranked_diseases:
        normalised_ranking = [_normalise(d) for d in ranked_diseases]
        for i, d in enumerate(normalised_ranking):
            d_tokens = _tokenise(d)
            gold_tokens = _tokenise(gold_disease)
            if d == gold_norm or _jaccard(d_tokens, gold_tokens) > 0.5:
                rank_position = i
                break

        if rank_position == 0:
            ranking_score = 0.30
        elif rank_position == 1:
            ranking_score = 0.15
        elif rank_position == 2:
            ranking_score = 0.05
    else:
        sel_norm = _normalise(selected_disease)
        sel_tokens = _tokenise(selected_disease)
        gold_tokens_set = _tokenise(gold_disease)
        if sel_norm == gold_norm or _jaccard(sel_tokens, gold_tokens_set) > 0.5:
            ranking_score = 0.30

    # Elimination reasoning quality (max 0.35)
    elim_score = 0.0
    elim_details: Dict[str, float] = {}
    if elimination_reasoning:
        for distractor in distractors:
            dist_norm = _normalise(distractor)
            reasoning_text = ""
            for key, val in elimination_reasoning.items():
                if _normalise(key) == dist_norm or _jaccard(_tokenise(key), _tokenise(distractor)) > 0.4:
                    reasoning_text = val
                    break

            if not reasoning_text:
                for _key, val in elimination_reasoning.items():
                    reasoning_text = val
                    break

            dist_score = 0.0
            if reasoning_text:
                reasoning_lower = reasoning_text.lower()
                if any(w in reasoning_lower for w in ["pathway", "network", "signal", "cascade"]):
                    dist_score += 0.04
                if any(w in reasoning_lower for w in ["gene", "mutation", "variant", "allele"]):
                    dist_score += 0.04
                if any(w in reasoning_lower for w in ["mechanism", "different", "instead", "rather",
                                                       "not associated", "unrelated", "does not"]):
                    dist_score += 0.04

            elim_details[dist_norm] = round(min(dist_score, 0.12), 4)
            elim_score += min(dist_score, 0.12)

    elim_score = min(elim_score, 0.35)

    # Supporting evidence quality (max 0.25)
    evidence_score = 0.0
    evidence_breakdown: Dict[str, Any] = {}
    if supporting_evidence and supporting_evidence.strip():
        steps = _extract_steps(supporting_evidence)
        step_count = len(steps)
        step_part = 0.05 if step_count >= 3 else (0.02 if step_count >= 1 else 0.0)

        gold_concepts = _extract_gene_names(gold_reasoning)
        pred_concepts = _extract_gene_names(supporting_evidence)
        if gold_concepts:
            concept_cov = len(pred_concepts & gold_concepts) / len(gold_concepts)
        else:
            concept_cov = 0.0
        concept_part = min(concept_cov, 1.0) * 0.10

        causal_count = _count_causal_connectors(supporting_evidence)
        causal_part = min(causal_count / 5.0, 1.0) * 0.05

        pathway_part = 0.0
        if pathway_genes:
            pathway_set = {g.upper() for g in pathway_genes}
            pathway_cov = len(pred_concepts & pathway_set) / len(pathway_set) if pathway_set else 0.0
            pathway_part = min(pathway_cov, 1.0) * 0.05

        evidence_score = step_part + concept_part + causal_part + pathway_part
        evidence_breakdown = {
            "step_count": step_count,
            "concept_coverage": round(concept_cov, 4),
            "causal_connectors": causal_count,
        }

    evidence_score = min(evidence_score, 0.25)

    # Logical consistency (max 0.10)
    consistency_score = 0.05
    if elimination_reasoning and supporting_evidence:
        sel_norm_check = _normalise(selected_disease)
        gold_tokens_check = _tokenise(gold_disease)
        sel_tokens_check = _tokenise(selected_disease)
        answer_is_correct = (
            sel_norm_check == gold_norm or
            _jaccard(sel_tokens_check, gold_tokens_check) > 0.5
        )
        if answer_is_correct:
            consistency_score = 0.08

        all_candidates_addressed = True
        if ranked_diseases and len(ranked_diseases) >= 4:
            consistency_score = min(consistency_score + 0.02, 0.10)
        elif elimination_reasoning and len(elimination_reasoning) >= len(distractors):
            consistency_score = min(consistency_score + 0.02, 0.10)

    total = ranking_score + elim_score + evidence_score + consistency_score

    breakdown = {
        "ranking_score": round(ranking_score, 4),
        "rank_position": rank_position,
        "elimination_score": round(elim_score, 4),
        "elimination_details": elim_details,
        "evidence_score": round(evidence_score, 4),
        "evidence_breakdown": evidence_breakdown,
        "consistency_score": round(consistency_score, 4),
    }
    return _clamp(total), breakdown


# =========================================================================
# Lab-mode graders (new for the hackathon "Drug Discovery Lab")
# =========================================================================


# --- Leaf-GO F1 -----------------------------------------------------------

def grade_leaf_go_f1(
    predicted_go_terms: Optional[List[str]],
    gold_leaf_text: str,
) -> Tuple[float, Dict[str, Any]]:
    """Leaf-only GO F1.

    The gold label is the concatenation of ``go_bp_leaf`` / ``go_mf_leaf``
    / ``go_cc_leaf`` from ``Protien_sft_reasoning.json`` — i.e. a small
    set of functionally discriminative terms rather than a flat list of
    every ancestor. This is a much sharper signal than the flat-set F1
    used by ``grade_protein_function``.

    Returns a score in [0.01, 0.99]. Empty gold → 0.50 (neutral).
    """
    gold_set = _extract_go_ids(gold_leaf_text or "")
    pred_set = _normalise_go_list(predicted_go_terms)

    if not gold_set:
        return _clamp(0.50), {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "note": "no gold leaf GO terms — neutral score",
        }
    if not pred_set:
        return _clamp(0.01), {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "note": "no predicted GO terms",
        }

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    score = 0.01 + f1 * 0.98
    return _clamp(score), {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": sorted(pred_set & gold_set),
        "missed": sorted(gold_set - pred_set),
    }


# --- Process-trace similarity --------------------------------------------

def _normalise_trace_line(text: str) -> str:
    """Normalise a reasoning step for cross-trace comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    # Keep GO:XXXXXX and IPRXXXXXX IDs, drop punctuation otherwise.
    text = re.sub(r"[^a-z0-9: \-]+", " ", text)
    return text.strip()


def grade_process_trace(
    predicted_steps: List[str],
    gold_steps: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """Stepwise similarity between the agent's reasoning trace and the gold CoT.

    Uses ``difflib.SequenceMatcher`` (deterministic, CPU-only, no GPU/API
    dependencies) after light normalisation that preserves biological IDs
    (GO:*, IPR*). Each gold step is matched to its best predicted step
    (greedy), and the score is the mean ratio.

    Score is in [0.01, 0.99]. This is the dense-per-step signal that makes
    the GRPO reward curve move within a short training budget.
    """
    gold_norm = [_normalise_trace_line(s) for s in (gold_steps or []) if s.strip()]
    pred_norm = [_normalise_trace_line(s) for s in (predicted_steps or []) if s.strip()]

    if not gold_norm:
        return _clamp(0.50), {
            "note": "no gold steps — neutral score",
            "pred_step_count": len(pred_norm),
            "gold_step_count": 0,
        }
    if not pred_norm:
        return _clamp(0.01), {
            "note": "no predicted steps",
            "pred_step_count": 0,
            "gold_step_count": len(gold_norm),
        }

    ratios: List[float] = []
    per_step: List[Dict[str, Any]] = []
    for gi, g in enumerate(gold_norm):
        best = 0.0
        best_pred_idx = -1
        for pi, p in enumerate(pred_norm):
            r = difflib.SequenceMatcher(None, g, p).ratio()
            if r > best:
                best = r
                best_pred_idx = pi
        ratios.append(best)
        per_step.append({"gold_idx": gi, "best_pred_idx": best_pred_idx, "ratio": round(best, 4)})

    mean_ratio = sum(ratios) / len(ratios)

    length_factor = min(len(pred_norm) / max(len(gold_norm), 1), 1.0)

    raw = 0.75 * mean_ratio + 0.25 * length_factor
    score = 0.01 + raw * 0.98
    return _clamp(score), {
        "mean_ratio": round(mean_ratio, 4),
        "length_factor": round(length_factor, 4),
        "pred_step_count": len(pred_norm),
        "gold_step_count": len(gold_norm),
        "per_step": per_step[:10],
    }


# --- Intervention plausibility -------------------------------------------

# Curated InterPro family-prefix / keyword -> allowed mode(s)-of-action.
# These are intentionally conservative: an intervention is marked
# "plausible" if its mode appears in the list for at least one matched
# family keyword of the target protein, and "off-target" otherwise.
_MOA_TABLE: List[Tuple[str, Set[str]]] = [
    ("kinase", {"inhibit", "degrade"}),
    ("phosphodiesterase", {"inhibit", "activate"}),
    ("phosphatase", {"inhibit", "activate"}),
    ("protease", {"inhibit"}),
    ("peptidase", {"inhibit"}),
    ("hydrolase", {"inhibit"}),
    ("oxidoreductase", {"inhibit", "activate"}),
    ("dehydrogenase", {"inhibit", "activate"}),
    ("transferase", {"inhibit"}),
    ("ligase", {"inhibit", "degrade"}),
    ("chaperone", {"activate", "chaperone", "upregulate"}),
    ("heat shock", {"activate", "chaperone", "upregulate"}),
    ("hsp", {"activate", "chaperone", "upregulate"}),
    ("chaperonin", {"activate", "chaperone", "upregulate"}),
    ("ubiquitin", {"inhibit", "degrade"}),
    ("e3", {"inhibit", "degrade"}),
    ("proteasome", {"inhibit"}),
    ("transcription factor", {"inhibit", "degrade", "upregulate"}),
    ("nuclear receptor", {"inhibit", "activate"}),
    ("receptor", {"inhibit", "activate"}),
    ("gpcr", {"inhibit", "activate"}),
    ("channel", {"inhibit", "activate"}),
    ("ion channel", {"inhibit", "activate"}),
    ("transporter", {"inhibit", "activate"}),
    ("sodium channel", {"inhibit", "activate"}),
    ("potassium channel", {"inhibit", "activate"}),
    ("calcium channel", {"inhibit", "activate"}),
    ("synuclein", {"degrade", "chaperone"}),
    ("scaffolding", {"degrade"}),
    ("scaffold", {"degrade"}),
    ("signal peptide", {"inhibit"}),
    ("sequestosome", {"upregulate", "activate"}),
    ("autophagy", {"upregulate", "activate"}),
    ("amyloid", {"degrade", "chaperone"}),
    ("tau", {"degrade", "chaperone"}),
    ("tdp-43", {"degrade", "chaperone"}),
    ("methyltransferase", {"inhibit"}),
    ("demethylase", {"inhibit"}),
    ("deacetylase", {"inhibit"}),
    ("acetyltransferase", {"inhibit"}),
    ("cytokine", {"inhibit", "activate"}),
    ("growth factor", {"inhibit", "activate"}),
    ("toxin", {"inhibit"}),
    ("antioxidant", {"upregulate", "activate"}),
]

_ALLOWED_MODES = {"inhibit", "activate", "degrade", "chaperone", "upregulate"}


def grade_intervention(
    proposal: Optional[Dict[str, str]],
    gold: ProteinSample,
    pathway_genes: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Plausibility score for a proposed ``{mode, target}`` intervention.

    The proposal is rewarded when:
    1. ``mode`` is one of the recognised MoAs.
    2. ``target`` matches either the gold protein / its pathway genes.
    3. The proposed ``mode`` is compatible with the target's functional
       class per the curated InterPro/keyword -> MoA lookup table.

    Returns a score in [0.01, 0.99].
    """
    if not proposal:
        return _clamp(0.01), {"note": "no intervention proposed"}

    mode_raw = (proposal.get("mode") or "").lower().strip()
    target_raw = (proposal.get("target") or "").strip()

    if not mode_raw or not target_raw:
        return _clamp(0.05), {"note": "proposal missing mode or target"}

    mode_score = 0.25 if mode_raw in _ALLOWED_MODES else 0.0

    # Target plausibility: match against gold protein id/name or pathway genes
    target_upper = target_raw.upper()
    target_score = 0.0
    if gold.protein_id and gold.protein_id.upper() in target_upper:
        target_score = 0.25
    else:
        name_tokens = _tokenise(gold.protein_names or "")
        prop_tokens = _tokenise(target_raw)
        if prop_tokens and name_tokens and _jaccard(prop_tokens, name_tokens) > 0.2:
            target_score = 0.20
        elif pathway_genes:
            genes_upper = {g.upper() for g in pathway_genes}
            if target_upper in genes_upper or any(g in target_upper for g in genes_upper):
                target_score = 0.20

    # MoA compatibility with target functional class
    haystack = " ".join([
        gold.protein_names or "",
        gold.protein_function or "",
        gold.interpro_formatted or "",
    ]).lower()

    matched_keywords: List[str] = []
    allowed_modes: Set[str] = set()
    for kw, modes in _MOA_TABLE:
        if kw in haystack:
            matched_keywords.append(kw)
            allowed_modes |= modes

    if allowed_modes:
        moa_score = 0.40 if mode_raw in allowed_modes else 0.05
    else:
        moa_score = 0.20  # unknown target class -> partial credit

    total = mode_score + target_score + moa_score

    return _clamp(total), {
        "mode": mode_raw,
        "target": target_raw,
        "mode_score": round(mode_score, 4),
        "target_score": round(target_score, 4),
        "moa_score": round(moa_score, 4),
        "matched_keywords": matched_keywords,
        "allowed_modes": sorted(allowed_modes),
    }


# --- Tool efficiency -----------------------------------------------------

def grade_tool_efficiency(
    tool_calls: List[Dict[str, Any]],
    predicted_reasoning: str,
    max_useful_calls: int = 6,
) -> Tuple[float, Dict[str, Any]]:
    """Reward the agent for calling tools whose results show up in its final reasoning.

    A tool call is ``useful`` when any token from its returned payload
    appears in the final reasoning text (and the call is unique). The
    final score rewards useful calls up to ``max_useful_calls`` and
    penalises redundant / unused calls.

    Score is in [0.01, 0.99].
    """
    if not tool_calls:
        return _clamp(0.50), {"note": "no tool calls"}

    reasoning_norm = _normalise(predicted_reasoning or "")
    reasoning_tokens = set(reasoning_norm.split())

    useful = 0
    redundant = 0
    seen: Set[str] = set()
    call_details: List[Dict[str, Any]] = []
    for call in tool_calls:
        name = call.get("tool_name", "unknown")
        args = call.get("tool_args") or {}
        result = call.get("result") or {}

        signature = name + ":" + ",".join(f"{k}={v}" for k, v in sorted(args.items()))
        is_redundant = signature in seen
        seen.add(signature)

        # Flatten the result and look for any distinctive token in the reasoning.
        payload = " ".join(str(v) for v in result.values() if isinstance(v, (str, int, float)))
        payload_tokens = set(_normalise(payload).split())
        distinctive = {t for t in payload_tokens if len(t) >= 4}
        hit = bool(distinctive & reasoning_tokens)

        if is_redundant:
            redundant += 1
        elif hit:
            useful += 1

        call_details.append({
            "tool": name,
            "args": args,
            "useful": hit and not is_redundant,
            "redundant": is_redundant,
        })

    useful_capped = min(useful, max_useful_calls)
    useful_ratio = useful_capped / max_useful_calls
    penalty = min(0.25, 0.05 * redundant)
    over_penalty = 0.0
    if len(tool_calls) > max_useful_calls + 4:
        over_penalty = min(0.15, 0.03 * (len(tool_calls) - max_useful_calls - 4))

    raw = 0.70 * useful_ratio + 0.30 - penalty - over_penalty
    return _clamp(raw), {
        "useful_calls": useful,
        "redundant_calls": redundant,
        "total_calls": len(tool_calls),
        "penalty": round(penalty, 4),
        "over_penalty": round(over_penalty, 4),
        "call_details": call_details[:15],
    }


# =========================================================================
# v2 graders: clinical diagnosis, perturbation QA, ligand design
# =========================================================================


def grade_clinical_diagnosis(
    predicted_answer: str,
    predicted_ranking: Optional[List[str]],
    predicted_reasoning: Optional[str],
    sample: "DiagnosisSample",
) -> Tuple[float, Dict[str, Any]]:
    """Grade a clinical differential-diagnosis submission.

    Blends:
      - 30% final diagnosis match (token Jaccard + exact bonus)
      - 25% ranking correctness (position of the gold final diagnosis)
      - 25% process-trace similarity to the gold gptoss120b CoT
      - 20% reasoning quality (token overlap + step structure)
    """
    # --- Final diagnosis match (max 0.30) ---
    gold_final = sample.final_diagnosis or ""
    gold_tokens = _tokenise(gold_final)
    pred_tokens = _tokenise(predicted_answer or "")
    dx_jaccard = _jaccard(pred_tokens, gold_tokens)
    if _normalise(predicted_answer or "") == _normalise(gold_final):
        dx_score = 0.30
    elif dx_jaccard > 0.5:
        dx_score = 0.18 + dx_jaccard * 0.12
    else:
        dx_score = dx_jaccard * 0.18

    # --- Ranking accuracy (max 0.25) ---
    rank_score = 0.0
    rank_position = -1
    if predicted_ranking:
        norm_rank = [_normalise(r) for r in predicted_ranking]
        gold_norm = _normalise(gold_final)
        for i, r in enumerate(norm_rank):
            if r == gold_norm or _jaccard(_tokenise(r), gold_tokens) > 0.5:
                rank_position = i
                break
        if rank_position == 0:
            rank_score = 0.25
        elif rank_position == 1:
            rank_score = 0.14
        elif rank_position == 2:
            rank_score = 0.06
    else:
        # Fall back to the final-diagnosis match only.
        if dx_score >= 0.18:
            rank_score = 0.10

    # --- Process-trace (max 0.25) ---
    pred_steps = _extract_steps(predicted_reasoning or "")
    process_raw, process_breakdown = grade_process_trace(pred_steps, sample.reasoning_steps)
    process_score = (process_raw - 0.01) / 0.98 * 0.25

    # --- Reasoning quality (max 0.20) ---
    reasoning_score = 0.0
    reasoning_details: Dict[str, Any] = {}
    if predicted_reasoning and predicted_reasoning.strip():
        gold_concepts = _tokenise(sample.raw_reasoning)
        pred_reasoning_tokens = _tokenise(predicted_reasoning)
        concept_cov = _jaccard(gold_concepts, pred_reasoning_tokens)
        reasoning_score += min(concept_cov, 1.0) * 0.12

        step_count = len(pred_steps)
        if step_count >= 4:
            reasoning_score += 0.08
        elif step_count >= 2:
            reasoning_score += 0.04

        reasoning_details = {
            "concept_coverage": round(concept_cov, 4),
            "step_count": step_count,
        }

    total = dx_score + rank_score + process_score + reasoning_score

    breakdown = {
        "dx_score": round(dx_score, 4),
        "dx_jaccard": round(dx_jaccard, 4),
        "rank_score": round(rank_score, 4),
        "rank_position": rank_position,
        "process_score": round(process_score, 4),
        "process_breakdown": process_breakdown,
        "reasoning_score": round(reasoning_score, 4),
        "reasoning_details": reasoning_details,
    }
    return _clamp(total), breakdown


def grade_perturbation_batch(
    predicted: Optional[Dict[str, bool]],
    gold: Dict[str, bool],
) -> Tuple[float, Dict[str, Any]]:
    """Grade a batch of binary CRISPRi answers.

    Reward = 0.5 * balanced_accuracy + 0.5 * macro_F1, clamped to
    [0.01, 0.99]. Missing answers count as a neutral 0.5 on that pair
    (counted as half right for balanced accuracy, excluded from F1 pools).
    """
    predicted = predicted or {}
    if not gold:
        return _clamp(0.50), {"note": "empty gold batch"}

    tp_yes = fp_yes = fn_yes = tn_yes = 0
    answered = 0
    correct = 0
    missing = 0
    per_pair: List[Dict[str, Any]] = []

    for pair_id, gold_answer in gold.items():
        if pair_id not in predicted:
            missing += 1
            per_pair.append({"pair_id": pair_id, "gold": gold_answer, "predicted": None, "correct": False})
            continue
        pred = bool(predicted[pair_id])
        answered += 1
        ok = pred == gold_answer
        if ok:
            correct += 1
        if gold_answer:
            if pred:
                tp_yes += 1
            else:
                fn_yes += 1
        else:
            if pred:
                fp_yes += 1
            else:
                tn_yes += 1
        per_pair.append({"pair_id": pair_id, "gold": gold_answer, "predicted": pred, "correct": ok})

    total = len(gold)
    if answered == 0:
        return _clamp(0.50 - 0.10), {
            "note": "no answers provided",
            "missing": missing,
            "total": total,
        }

    # Macro F1 between yes/no classes.
    prec_yes = tp_yes / (tp_yes + fp_yes) if (tp_yes + fp_yes) else 0.0
    rec_yes = tp_yes / (tp_yes + fn_yes) if (tp_yes + fn_yes) else 0.0
    f1_yes = (2 * prec_yes * rec_yes / (prec_yes + rec_yes)) if (prec_yes + rec_yes) else 0.0

    prec_no = tn_yes / (tn_yes + fn_yes) if (tn_yes + fn_yes) else 0.0
    rec_no = tn_yes / (tn_yes + fp_yes) if (tn_yes + fp_yes) else 0.0
    f1_no = (2 * prec_no * rec_no / (prec_no + rec_no)) if (prec_no + rec_no) else 0.0
    macro_f1 = (f1_yes + f1_no) / 2

    # Balanced accuracy.
    pos_total = tp_yes + fn_yes
    neg_total = tn_yes + fp_yes
    sens = tp_yes / pos_total if pos_total else 0.0
    spec = tn_yes / neg_total if neg_total else 0.0
    balanced_acc = (sens + spec) / 2

    # Penalty for missing answers (neutral 0.5 on each, so missing pulls toward 0.5).
    coverage = answered / total
    raw = (0.5 * macro_f1 + 0.5 * balanced_acc) * coverage + 0.5 * (1 - coverage)

    breakdown = {
        "answered": answered,
        "missing": missing,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / answered, 4) if answered else 0.0,
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "f1_yes": round(f1_yes, 4),
        "f1_no": round(f1_no, 4),
        "per_pair": per_pair[:15],
    }
    return _clamp(raw), breakdown


# =========================================================================
# Ligand-design grading (no rdkit dependency)
# =========================================================================


_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|Cl|Br|[BCNOSPFIcnosp]|[=\#\-\+\/\\\(\)\.]|\d)"
)

# Pattern for SELFIES-style ``[mol] ... [/mol]`` fragments found in
# drug_discovery_hetionet.json. We strip these wrappers for tokenisation.
_SELFIES_WRAP_RE = re.compile(r"\[/?mol\]", re.IGNORECASE)


def _tokenise_smiles(smiles: str) -> List[str]:
    """Pure-python SMILES / SELFIES tokenizer.

    Splits on bracketed atoms (``[C@H]``), two-letter atoms (``Cl``,
    ``Br``), bond symbols, parens, ring digits, and single-letter atoms.
    Falls back to character-level for anything unrecognised so that
    SELFIES-style blocks (``[O]``, ``[Branch1]``, ...) still produce a
    sensible token stream.
    """
    if not smiles:
        return []
    clean = _SELFIES_WRAP_RE.sub(" ", smiles)
    tokens: List[str] = []
    idx = 0
    while idx < len(clean):
        ch = clean[idx]
        if ch.isspace():
            idx += 1
            continue
        match = _SMILES_TOKEN_RE.match(clean, idx)
        if match:
            tokens.append(match.group(0))
            idx = match.end()
        else:
            tokens.append(ch)
            idx += 1
    return tokens


def _smiles_token_set(smiles: str) -> Set[str]:
    return {t.lower() for t in _tokenise_smiles(smiles)}


def grade_ligand_match(
    predicted_ligand: Optional[str],
    sample: "LigandSample",
    top1000: Optional[List["DrugRecord"]] = None,
    top1000_by_smiles: Optional[Dict[str, "DrugRecord"]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Grade a proposed ligand (SMILES or named drug) against the gold target.

    Blended score:
      - 40% SMILES token Jaccard (case-insensitive, SELFIES-aware).
      - 25% named-drug exact match (when gold is a name) OR SMILES equality bonus.
      - 25% catalogue membership bonus weighted by drug_score.
      - 10% property proximity (logP, num_atoms) vs. the gold molecule.
    """
    top1000 = top1000 or []
    if top1000_by_smiles is None:
        top1000_by_smiles = {d.smiles: d for d in top1000}

    if not predicted_ligand or not predicted_ligand.strip():
        return _clamp(0.01), {"note": "no predicted ligand"}

    pred_clean = predicted_ligand.strip()
    gold_target = (sample.gold_target or "").strip()

    # 1. SMILES token Jaccard.
    pred_tokens = _smiles_token_set(pred_clean)
    gold_tokens = _smiles_token_set(gold_target)
    smiles_jaccard = _jaccard(pred_tokens, gold_tokens)
    jaccard_score = smiles_jaccard * 0.40

    # 2. Named-drug exact match (or SMILES equality bonus when gold is SMILES).
    name_score = 0.0
    if not sample.gold_is_smiles:
        if _normalise(pred_clean) == _normalise(gold_target) and gold_target:
            name_score = 0.25
        else:
            name_score = _jaccard(_tokenise(pred_clean), _tokenise(gold_target)) * 0.15
    else:
        if pred_clean == gold_target:
            name_score = 0.25
        elif smiles_jaccard > 0.95:
            name_score = 0.20

    # 3. Catalogue membership weighted by drug_score.
    catalogue_score = 0.0
    matched_record: Optional[DrugRecord] = top1000_by_smiles.get(pred_clean)
    if matched_record is not None:
        # Drug scores in the dataset range roughly 9 - 11, so normalise softly.
        norm = max(0.0, min(1.0, (matched_record.drug_score - 5.0) / 10.0))
        catalogue_score = 0.10 + norm * 0.15

    # 4. Property proximity vs. the gold-SMILES catalogue entry.
    prop_score = 0.0
    gold_record = top1000_by_smiles.get(gold_target) if sample.gold_is_smiles else None
    if gold_record is not None and matched_record is not None:
        atom_diff = abs(matched_record.num_atoms - gold_record.num_atoms) / max(gold_record.num_atoms, 1)
        atom_close = max(0.0, 1.0 - atom_diff)
        logp_diff = abs(matched_record.logp - gold_record.logp)
        logp_close = max(0.0, 1.0 - logp_diff / 5.0)
        prop_score = (atom_close * 0.5 + logp_close * 0.5) * 0.10
    elif matched_record is not None:
        # Some credit for producing a recognisable drug-like molecule.
        prop_score = 0.05

    total = jaccard_score + name_score + catalogue_score + prop_score
    breakdown = {
        "smiles_jaccard": round(smiles_jaccard, 4),
        "jaccard_score": round(jaccard_score, 4),
        "name_score": round(name_score, 4),
        "catalogue_score": round(catalogue_score, 4),
        "prop_score": round(prop_score, 4),
        "in_catalogue": matched_record is not None,
        "gold_is_smiles": sample.gold_is_smiles,
    }
    return _clamp(total), breakdown


def grade_drug_design_phase(
    predicted_ligand: Optional[str],
    sample: "LigandSample",
    drug_tool_calls: List[Dict[str, Any]],
    predicted_reasoning: str,
    top1000_by_smiles: Optional[Dict[str, "DrugRecord"]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Grader for the DRUG_DESIGN addon inside an existing lab episode.

    Combines ``grade_ligand_match`` with ``grade_tool_efficiency`` scoped to
    drug-discovery tool calls so the composite sits in [0.01, 0.99].
    """
    ligand_score, ligand_breakdown = grade_ligand_match(
        predicted_ligand, sample, top1000_by_smiles=top1000_by_smiles,
    )
    tool_score, tool_breakdown = grade_tool_efficiency(
        drug_tool_calls, predicted_reasoning, max_useful_calls=3,
    )

    blended = 0.75 * ligand_score + 0.25 * tool_score
    return _clamp(blended), {
        "ligand": ligand_breakdown,
        "ligand_score": round(ligand_score, 4),
        "tool": tool_breakdown,
        "tool_score": round(tool_score, 4),
    }


# =========================================================================
# v3 graders: KEGG pathway reasoning, directional perturbation, umbrella
# =========================================================================


_PATHWAY_OP_RE = re.compile(r"-\||->|//|==|=>|\+")
_PATHWAY_SPLIT_RE = re.compile(r"[,\s\(\)\[\]]+")


def _parse_pathway_tokens(graph: str) -> List[str]:
    """Tokenise a KEGG declarative pathway string into atoms + operators.

    The upstream format uses ``->``, ``-|``, ``//``, ``==``, ``=>``, ``+``
    as operators, and comma-separated targets inside parentheses, e.g.
    ``TARDBP* -| CxI -> Q`` or ``PSAP* // (GBA,GALC)``. Tokens are kept
    uppercase and the trailing ``*`` (perturbation marker) is preserved so
    agents who quote the graph verbatim get full credit.
    """
    if not graph:
        return []
    # Insert spaces around operators to make splitting easy.
    replaced = _PATHWAY_OP_RE.sub(lambda m: f" {m.group(0)} ", graph)
    raw = _PATHWAY_SPLIT_RE.split(replaced)
    tokens: List[str] = []
    for tok in raw:
        t = tok.strip()
        if not t:
            continue
        tokens.append(t.upper())
    return tokens


def grade_kegg_reasoning(
    predicted_answer: str,
    predicted_reasoning: str,
    predicted_mentioned_genes: Optional[List[str]],
    sample: "KeggSample",
) -> Tuple[float, Dict[str, Any]]:
    """Grade a KEGG pathway-networked variant-to-disease submission.

    Blends four signals:

    - 30% disease-name match (reuses :py:func:`grade_dna_classification`).
    - 25% pathway-graph fidelity: Jaccard between tokens of the gold graph
      and tokens the agent echoed anywhere in its reasoning. Rewards
      agents that _quote the graph_ (tokens include operators like
      ``-|`` / ``->``).
    - 25% process-trace similarity to the gold stepwise reasoning.
    - 20% gene-coverage F1 between ``predicted_mentioned_genes`` (or gene
      symbols extracted from the reasoning) and the sample's pathway gene
      list.
    """
    # 1) Disease accuracy (30%).
    disease_score, disease_bd = grade_dna_classification(predicted_answer or "", sample.answer or "")
    disease_component = disease_score * 0.30

    # 2) Pathway-graph fidelity (25%).
    gold_tokens = set(_parse_pathway_tokens(sample.pathway_graph or ""))
    reasoning_tokens_upper: Set[str] = set()
    if predicted_reasoning:
        for raw in re.split(r"\s+", predicted_reasoning):
            raw_u = raw.strip().upper().rstrip(".,;:")
            if raw_u:
                reasoning_tokens_upper.add(raw_u)
        # Also tokenise any graph-like fragments the agent wrote.
        reasoning_tokens_upper |= set(_parse_pathway_tokens(predicted_reasoning))
    if gold_tokens:
        graph_jaccard = len(gold_tokens & reasoning_tokens_upper) / len(gold_tokens | reasoning_tokens_upper)
    else:
        graph_jaccard = 0.0
    graph_component = graph_jaccard * 0.25

    # 3) Process-trace similarity (25%).
    pred_steps = _extract_steps(predicted_reasoning or "")
    process_raw, process_bd = grade_process_trace(pred_steps, sample.reasoning_steps)
    process_component = (process_raw - 0.01) / 0.98 * 0.25

    # 4) Pathway-gene coverage F1 (20%).
    gold_genes = {g.upper() for g in (sample.genes_in_pathway or []) if g}
    pred_genes: Set[str] = set()
    if predicted_mentioned_genes:
        pred_genes |= {g.strip().upper() for g in predicted_mentioned_genes if g and g.strip()}
    if predicted_reasoning:
        pred_genes |= {g.upper() for g in _extract_gene_names(predicted_reasoning)}
    if gold_genes and pred_genes:
        tp = len(pred_genes & gold_genes)
        precision = tp / len(pred_genes)
        recall = tp / len(gold_genes)
        gene_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    else:
        precision = recall = gene_f1 = 0.0
    gene_component = gene_f1 * 0.20

    total = disease_component + graph_component + process_component + gene_component

    breakdown = {
        "disease": {"score": round(disease_score, 4), "component": round(disease_component, 4), **disease_bd},
        "graph_jaccard": round(graph_jaccard, 4),
        "graph_component": round(graph_component, 4),
        "process_score": round(process_raw, 4),
        "process_component": round(process_component, 4),
        "process_breakdown": process_bd,
        "gene_precision": round(precision, 4),
        "gene_recall": round(recall, 4),
        "gene_f1": round(gene_f1, 4),
        "gene_component": round(gene_component, 4),
        "gold_graph_tokens": sorted(gold_tokens),
    }
    return _clamp(total), breakdown


_DIRECTION_MAP: Dict[str, str] = {
    "increase": "Increase",
    "increased": "Increase",
    "up": "Increase",
    "upregulate": "Increase",
    "upregulated": "Increase",
    "+": "Increase",
    "positive": "Increase",
    "yes": "Increase",
    "true": "Increase",
    "decrease": "Decrease",
    "decreased": "Decrease",
    "down": "Decrease",
    "downregulate": "Decrease",
    "downregulated": "Decrease",
    "-": "Decrease",
    "negative": "Decrease",
    "no": "Decrease",
    "false": "Decrease",
    "unknown": "Unknown",
    "neutral": "Unknown",
    "unchanged": "Unknown",
    "unsure": "Unknown",
    "none": "Unknown",
}


def _normalise_direction(label: Optional[str]) -> str:
    """Map free-form agent output to one of {Increase, Decrease, Unknown}."""
    if label is None:
        return "Unknown"
    text = str(label).strip().lower()
    if not text:
        return "Unknown"
    if text in _DIRECTION_MAP:
        return _DIRECTION_MAP[text]
    # Fall back to substring search for phrases like "probably increase".
    for key, val in _DIRECTION_MAP.items():
        if len(key) >= 3 and key in text:
            return val
    return "Unknown"


def grade_perturbation_direction(
    predicted: Optional[Dict[str, str]],
    gold: Dict[str, str],
) -> Tuple[float, Dict[str, Any]]:
    """Grade a 3-class directional CRISPRi batch.

    Each gold entry is one of ``Increase`` / ``Decrease``. Predictions are
    normalised via :func:`_normalise_direction` before scoring. Missing or
    ``Unknown`` answers count as a neutral 0.33 on that pair.

    Score = 0.5 * balanced accuracy + 0.5 * macro-F1 (over 3 classes),
    clamped to [0.01, 0.99].
    """
    predicted = predicted or {}
    classes = ("Increase", "Decrease", "Unknown")
    if not gold:
        return _clamp(0.50), {"note": "empty gold batch"}

    # Confusion counts per class (true -> predicted).
    confusion: Dict[str, Dict[str, int]] = {c: {d: 0 for d in classes} for c in classes}
    correct = 0
    answered = 0
    missing = 0
    unknown_hits = 0
    per_pair: List[Dict[str, Any]] = []
    total = 0

    for pair_id, gold_raw in gold.items():
        g = _normalise_direction(gold_raw)
        if g not in ("Increase", "Decrease"):
            # Skip malformed gold rows (shouldn't happen for pert_dir).
            continue
        total += 1
        raw_pred = predicted.get(pair_id)
        if raw_pred is None:
            p = "Unknown"
            missing += 1
        else:
            p = _normalise_direction(raw_pred)
            answered += 1
        if p == "Unknown":
            unknown_hits += 1
        confusion[g][p] += 1
        if p == g:
            correct += 1
        per_pair.append({
            "pair_id": pair_id,
            "gold": g,
            "predicted": p,
            "correct": p == g,
        })

    if total == 0:
        return _clamp(0.50), {"note": "no classifiable gold rows"}

    # Class-level precision / recall over Increase + Decrease (the two
    # classes that carry information). Unknown is excluded from precision
    # but counts against recall.
    def _prec_rec(c: str) -> Tuple[float, float]:
        tp = confusion[c][c]
        fp = sum(confusion[other][c] for other in classes if other != c)
        fn = sum(confusion[c][other] for other in classes if other != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return precision, recall

    f1_components: Dict[str, float] = {}
    recalls: List[float] = []
    for c in ("Increase", "Decrease"):
        p, r = _prec_rec(c)
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        f1_components[c] = round(f1, 4)
        recalls.append(r)

    macro_f1 = sum(f1_components.values()) / len(f1_components)
    balanced_acc = sum(recalls) / len(recalls)
    # Missing pair penalty — neutral 0.33 each, interpolate toward the
    # blended macro score.
    coverage = (total - unknown_hits) / total if total else 0.0
    raw = (0.5 * macro_f1 + 0.5 * balanced_acc) * coverage + 0.33 * (1 - coverage)

    breakdown = {
        "answered": answered,
        "missing": missing,
        "unknown": unknown_hits,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "f1_per_class": f1_components,
        "confusion": confusion,
        "per_pair": per_pair[:15],
    }
    return _clamp(raw), breakdown


def grade_perturbation_benchmark(
    predicted: Optional[Dict[str, str]],
    gold_by_variant: Dict[str, Dict[str, str]],
) -> Tuple[float, Dict[str, Any]]:
    """Umbrella grader for ``perturbation_benchmark``.

    Gold payload is a dict keyed by variant name
    (``pert_dir``/``pert_de``/``gse_pert``/``gse_gene``), each mapping
    ``pair_id -> expected string``. We reuse
    :py:func:`grade_perturbation_direction` per variant and return the
    weighted mean (25% per variant). Variants without any gold entries
    contribute a neutral 0.33.
    """
    predicted = predicted or {}
    variants = ("pert_dir", "pert_de", "gse_pert", "gse_gene")
    sub_scores: Dict[str, float] = {}
    sub_breakdowns: Dict[str, Dict[str, Any]] = {}

    for variant in variants:
        gold_sub = gold_by_variant.get(variant) or {}
        if not gold_sub:
            sub_scores[variant] = 0.33
            sub_breakdowns[variant] = {"note": "no gold entries"}
            continue
        pred_sub = {pid: predicted[pid] for pid in gold_sub if pid in predicted}
        score, bd = grade_perturbation_direction(pred_sub, gold_sub)
        sub_scores[variant] = score
        sub_breakdowns[variant] = bd

    weighted = sum(sub_scores[v] for v in variants) / len(variants)
    breakdown = {
        "per_variant": {v: round(sub_scores[v], 4) for v in variants},
        "weights": {v: 0.25 for v in variants},
        "breakdowns": sub_breakdowns,
    }
    return _clamp(weighted), breakdown
