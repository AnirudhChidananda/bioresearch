"""
Grading functions for the Bioresearch Environment.

Each grader returns (score, breakdown_dict) where score is in [0.01, 0.99].
Grading is deterministic: same inputs always produce the same score.
Designed for GRPO compatibility with smooth, continuous reward signals
and step-level reasoning decomposition.
"""

import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .data_loader import ProteinSample
except ImportError:
    from server.data_loader import ProteinSample


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
