"""
Data loading and sampling for the Bioresearch Environment.

Handles the HuggingFace dataset JSON format (features + rows),
provides typed dataclass access, and supports GRPO-compatible
deterministic same-prompt replay via task_id lookups.

The loader owns three datasets:
    - ``DNA_reasoning.json``          — DNA mutation → disease samples.
    - ``Protien_sft_reasoning.json``  — protein sequence → function
      samples, including gold ``reasoning`` chains and ``go_pred_leaf``
      labels used by the new leaf-GO grader.
    - ``Protien_catalogue.json``      — SFT model <think> traces used
      as the gold chain-of-thought for the process-reward grader and
      the Theme-4 curriculum self-play task.

Filenames intentionally preserve the upstream ``Protien_`` spelling
(a misspelling in the source dataset) to avoid a git rename mid-
hackathon. The typo is tracked in ``knowledgebase/improvement.md``.
"""

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================
# Dataclasses
# =========================================================================


@dataclass
class DNASample:
    row_idx: int
    question: str
    answer: str
    reasoning: str
    reference_sequence: str
    variant_sequence: str


@dataclass
class ProteinSample:
    row_idx: int
    protein_id: str
    protein_names: str
    protein_function: str
    organism: str
    length: float
    subcellular_location: str
    sequence: str
    go_ids: List[str] = field(default_factory=list)
    go_bp: List[str] = field(default_factory=list)
    go_mf: List[str] = field(default_factory=list)
    go_cc: List[str] = field(default_factory=list)
    interpro_ids: List[str] = field(default_factory=list)
    interpro_formatted: str = ""
    ppi_formatted: str = ""
    go_pred: str = ""

    # New fields from Protien_sft_reasoning.json
    reasoning: str = ""
    final_answer: str = ""
    go_pred_leaf: str = ""
    go_bp_leaf: str = ""
    go_mf_leaf: str = ""
    go_cc_leaf: str = ""
    interaction_partners: List[str] = field(default_factory=list)
    interpro_location: str = ""
    structure_path: str = ""
    string_id: str = ""


@dataclass
class CatalogueSample:
    """A record from ``Protien_catalogue.json`` containing an SFT <think> trace."""

    row_idx: int
    protein_id: str
    sequence: str
    organism: str
    think_steps: List[str]
    structured_answer: str
    raw_generation: str


@dataclass
class DiagnosisSample:
    """A radiology case with differential diagnosis and gold CoT steps.

    Source: ``diagnosis_training_data.json``. ``reasoning_steps`` is parsed
    from the gptoss120b step-wise explanation and used by the process-trace
    grader for dense per-step reward.
    """

    row_idx: int
    case_id: str
    description: str
    differentials: List[str]
    final_diagnosis: str
    reasoning_steps: List[str]
    raw_reasoning: str


@dataclass
class PerturbationSample:
    """A binary CRISPRi perturbation Q&A pair from ``PertubationQA_language_pert_de.json``."""

    row_idx: int
    pair_id: str
    query_gene: str
    target_gene: str
    cell_line: str
    question: str
    answer: bool  # True == "Yes", False == "No"


@dataclass
class LigandSample:
    """A gene -> drug supervision pair from ``drug_discovery_hetionet.json``.

    ``gold_target`` is either a SELFIES-style ``[mol] ... [/mol]`` block
    (``gold_is_smiles=True``) or a drug name (``gold_is_smiles=False``).
    """

    row_idx: int
    gene: str
    go_neighbors_text: str
    gold_target: str
    gold_is_smiles: bool
    prompt: str


@dataclass
class DrugRecord:
    """A high-pIC50 small molecule from ``SMILES_top1000_drug_discovery.json``."""

    smiles: str
    pic50: float
    num_atoms: int
    logp: float
    drug_score: float


@dataclass
class KeggSample:
    """KEGG pathway-networked variant-to-disease reasoning sample.

    Source: ``kegg_reasoning.json`` / ``kegg_reasoning_2.json``. Each sample
    is a DNA variant whose ``question`` embeds a declarative pathway graph
    (``TARDBP* -| CxI -> Q``) plus a gene list. The grader rewards agents
    for quoting and reasoning over the pathway graph explicitly.
    """

    row_idx: int
    case_id: str
    question: str
    pathway_graph: str
    genes_in_pathway: List[str]
    answer: str
    reasoning_steps: List[str]
    raw_reasoning: str
    reference_sequence: str
    variant_sequence: str


@dataclass
class PerturbationDirSample:
    """A single directional / differential / gene-set CRISPRi QA sample.

    ``variant`` labels which of the four upstream ``PertubationQA_language_*``
    files this sample came from:

    - ``pert_dir``  — "Would you expect ... to increase or decrease ...?"
    - ``pert_de``   — "Would you expect ... to impact ...?" (binary Yes/No)
    - ``gse_pert``  — "What other genes might exert similar effects ...?"
    - ``gse_gene``  — "What other genes might respond similarly ...?"
    """

    row_idx: int
    pair_id: str
    query_gene: str
    target_gene: str
    cell_line: str
    question: str
    answer: str
    variant: str


# =========================================================================
# Helpers
# =========================================================================


def _find_data_dir() -> Path:
    """Locate the data/ directory relative to common project roots."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data",
        Path(os.getcwd()) / "data",
        Path("/app/env/data"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Cannot find data/ directory. Searched: {candidates}")


def _pick_existing(data_path: Path, names: List[str]) -> Optional[Path]:
    """Return the first path whose basename exists in ``data_path``."""
    for n in names:
        p = data_path / n
        if p.is_file():
            return p
    return None


_STEP_SPLIT_RE = re.compile(
    r"(?:^|\n)(?:Step\s*\d+\s*[:.]|\d+\s*[:.)])",
    re.MULTILINE | re.IGNORECASE,
)

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n", re.MULTILINE)


def _extract_think_steps(raw_generation: str) -> Tuple[List[str], str]:
    """Split the SFT generation into (ordered_think_steps, structured_answer).

    The catalogue generations have shape::

        <think>
        ...paragraph 1...

        ...paragraph 2...
        </think>

        - Functional Summary: ...
        - UniProt Summary: ...
        - InterPro: ...

    We treat each paragraph within ``<think>`` as a step. If the generation
    lacks explicit ``<think>`` tags the whole body is treated as a single
    step and ``structured_answer`` is the full text.
    """
    if not raw_generation:
        return [], ""

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", raw_generation, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_body = think_match.group(1).strip()
        structured = raw_generation[think_match.end():].strip()
    else:
        think_body = raw_generation.strip()
        structured = raw_generation.strip()

    steps = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(think_body) if p.strip()]
    if not steps and think_body:
        steps = [think_body]
    return steps, structured


_DIAG_STEP_RE = re.compile(
    r"(?:^|\n)\s*(?:Step\s*\d+|\d+\s*[\.\)])\s*[–\-:\.]",
    re.IGNORECASE,
)


def _extract_diagnosis_steps(reasoning: str) -> List[str]:
    """Split a gptoss120b reasoning trace into ordered steps.

    The upstream format mostly uses ``Step N – ...`` headers. Falls back to
    paragraph splits when no explicit step headers are found. Markdown tables
    and extra whitespace are preserved on purpose (the process-trace grader
    benefits from the extra tokens).
    """
    if not reasoning:
        return []
    positions: List[int] = [m.start() for m in _DIAG_STEP_RE.finditer(reasoning)]
    if positions:
        positions.append(len(reasoning))
        steps = []
        for i in range(len(positions) - 1):
            chunk = reasoning[positions[i]: positions[i + 1]].strip()
            if chunk:
                steps.append(chunk)
        if steps:
            return steps
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(reasoning) if p.strip()]


_DIFFERENTIAL_SPLIT_RE = re.compile(r"\s*,\s*")


def _split_differentials(text: str) -> List[str]:
    """Parse a ``DifferentialDiagnosisList`` field into clean strings."""
    if not text:
        return []
    items = [t.strip(" .") for t in _DIFFERENTIAL_SPLIT_RE.split(text)]
    return [t for t in items if t]


# =========================================================================
# DataLoader
# =========================================================================


class DataLoader:
    """Loads and manages bioresearch datasets with GRPO-compatible access patterns."""

    EPISODE_SPLIT = 80  # first 80 samples for episodes, last 20 for baseline
    NOTEBOOK_CHAR_CAP = 400  # max chars returned per tool call (to keep prompts short)

    def __init__(self, data_dir: Optional[str] = None):
        data_path = Path(data_dir) if data_dir else _find_data_dir()

        dna_path = data_path / "DNA_reasoning.json"
        protein_path = _pick_existing(
            data_path,
            ["Protien_sft_reasoning.json", "Protein_sft_reasoning.json", "Protein_reasoning.json", "Protein_data.json"],
        )
        if protein_path is None:
            raise FileNotFoundError(
                f"No protein dataset found in {data_path}. Expected one of "
                "Protien_sft_reasoning.json / Protein_data.json."
            )
        catalogue_path = _pick_existing(
            data_path,
            ["Protien_catalogue.json", "Protein_catalogue.json"],
        )

        # New (optional) datasets for v2 tasks.
        bridge_path = _pick_existing(data_path, ["protein_catalogue_bridge.json"])
        diagnosis_path = _pick_existing(data_path, ["diagnosis_training_data.json"])
        perturbation_path = _pick_existing(data_path, ["PertubationQA_language_pert_de.json", "PertubationQA_language_pert_de.json"])
        hetionet_path = _pick_existing(data_path, ["drug_discovery_hetionet.json"])
        top1000_path = _pick_existing(data_path, ["SMILES_top1000_drug_discovery.json"])

        # v3 additive pools.
        protein_data_2_path = _pick_existing(data_path, ["Protien_data_2.json", "Protein_data_2.json"])
        protein_sft_2_path = _pick_existing(data_path, ["Protien_sft_reasoning_2.json", "Protein_sft_reasoning_2.json"])
        kegg_paths = [
            p for p in (
                data_path / "kegg_reasoning.json",
                data_path / "kegg_reasoning_2.json",
            ) if p.is_file()
        ]
        pert_variant_paths = {
            "pert_dir": _pick_existing(data_path, ["PertubationQA_language_pert_dir.json", "PertubationQA_Language_pert_dir.json"]),
            "pert_de": _pick_existing(data_path, ["PertubationQA_language_pert_de.json", "PertubationQA_Language_pert_de.json"]),
            "gse_pert": _pick_existing(data_path, ["PertubationQA_language_gse_pert.json", "PertubationQA_Language_gse_pert.json"]),
            "gse_gene": _pick_existing(data_path, ["PertubationQA_language_gse_gene.json", "PertubationQA_Language_gse_gene.json"]),
        }

        self._dna_samples: List[DNASample] = self._load_dna(dna_path)
        self._protein_samples: List[ProteinSample] = self._load_protein(protein_path)
        self._catalogue_samples: List[CatalogueSample] = (
            self._load_catalogue(catalogue_path) if catalogue_path else []
        )
        self._bridge_records: List[ProteinSample] = (
            self._load_catalogue_bridge(bridge_path) if bridge_path else []
        )
        self._diagnosis_samples: List[DiagnosisSample] = (
            self._load_diagnosis(diagnosis_path) if diagnosis_path else []
        )
        self._perturbation_samples: List[PerturbationSample] = (
            self._load_perturbation(perturbation_path) if perturbation_path else []
        )
        self._ligand_samples: List[LigandSample] = (
            self._load_hetionet(hetionet_path) if hetionet_path else []
        )
        self._top1000: List[DrugRecord] = (
            self._load_top1000(top1000_path) if top1000_path else []
        )

        # v3: merge the secondary protein pools by dedup on protein_id. Any
        # v2 row whose protein_id collides with the v1 pool is discarded; the
        # remaining rows are re-indexed to offset +1000 so task_ids never
        # collide with existing ``protein_###`` IDs.
        if protein_data_2_path or protein_sft_2_path:
            existing_pids = {s.protein_id for s in self._protein_samples if s.protein_id}
            merged_v2: List[ProteinSample] = []
            v2_ids: set = set()
            # Preferred source: the richer SFT reasoning pool.
            if protein_sft_2_path:
                for s in self._load_protein(protein_sft_2_path):
                    if not s.protein_id or s.protein_id in existing_pids or s.protein_id in v2_ids:
                        continue
                    v2_ids.add(s.protein_id)
                    merged_v2.append(s)
            # Second source: the raw catalogue v2 — fills gaps.
            if protein_data_2_path:
                for s in self._load_protein(protein_data_2_path):
                    if not s.protein_id or s.protein_id in existing_pids or s.protein_id in v2_ids:
                        continue
                    v2_ids.add(s.protein_id)
                    merged_v2.append(s)
            # Offset row_idx so task_ids in the merged pool never collide.
            for i, s in enumerate(merged_v2):
                s.row_idx = 1000 + i
            self._protein_samples.extend(merged_v2)

        # v3: KEGG pathway reasoning pool (kegg_reasoning.json + _2).
        self._kegg_samples: List[KeggSample] = []
        offset = 0
        for kp in kegg_paths:
            loaded = self._load_kegg(kp, offset=offset)
            self._kegg_samples.extend(loaded)
            offset += len(loaded)

        # v3: four directional / GSE perturbation variants.
        self._pert_variants: Dict[str, List[PerturbationDirSample]] = {}
        for variant_name, variant_path in pert_variant_paths.items():
            if variant_path is not None:
                self._pert_variants[variant_name] = self._load_pert_variant(variant_path, variant_name)
            else:
                self._pert_variants[variant_name] = []

        self._dna_by_id: Dict[str, DNASample] = {
            f"dna_{s.row_idx:03d}": s for s in self._dna_samples
        }
        self._protein_by_id: Dict[str, ProteinSample] = {
            f"protein_{s.row_idx:03d}": s for s in self._protein_samples
        }
        self._catalogue_by_id: Dict[str, CatalogueSample] = {
            f"catalogue_{s.row_idx:03d}": s for s in self._catalogue_samples
        }
        self._protein_by_uniprot: Dict[str, ProteinSample] = {
            s.protein_id: s for s in self._protein_samples if s.protein_id
        }
        self._catalogue_by_uniprot: Dict[str, CatalogueSample] = {
            s.protein_id: s for s in self._catalogue_samples if s.protein_id
        }
        self._bridge_by_uniprot: Dict[str, ProteinSample] = {
            s.protein_id: s for s in self._bridge_records if s.protein_id
        }

        self._diagnosis_by_id: Dict[str, DiagnosisSample] = {
            f"diagnosis_{s.row_idx:03d}": s for s in self._diagnosis_samples
        }
        self._perturbation_by_id: Dict[str, PerturbationSample] = {
            s.pair_id: s for s in self._perturbation_samples
        }
        self._ligand_by_id: Dict[str, LigandSample] = {
            f"ligand_{s.row_idx:03d}": s for s in self._ligand_samples
        }
        self._top1000_by_smiles: Dict[str, DrugRecord] = {
            d.smiles: d for d in self._top1000
        }

        self._kegg_by_id: Dict[str, KeggSample] = {
            f"kegg_{s.row_idx:04d}": s for s in self._kegg_samples
        }
        # Flat index of every directional-variant sample by pair_id.
        self._pert_dir_by_id: Dict[str, PerturbationDirSample] = {}
        for variant_name, samples in self._pert_variants.items():
            for s in samples:
                self._pert_dir_by_id[s.pair_id] = s

        # Gene -> ligand index for get_candidate_ligands (deterministic order).
        self._ligands_by_gene: Dict[str, List[LigandSample]] = {}
        for s in self._ligand_samples:
            self._ligands_by_gene.setdefault(s.gene.upper(), []).append(s)

        self._all_disease_answers: List[str] = sorted({
            s.answer.lower().strip() for s in self._dna_samples
        })

    # -- Loading -----------------------------------------------------------

    @staticmethod
    def _load_dna(path: Path) -> List[DNASample]:
        with open(path, "r") as f:
            data = json.load(f)
        samples = []
        for entry in data["rows"]:
            row = entry["row"]
            samples.append(DNASample(
                row_idx=entry["row_idx"],
                question=row["question"],
                answer=row["answer"],
                reasoning=row["reasoning"],
                reference_sequence=row["reference_sequence"],
                variant_sequence=row["variant_sequence"],
            ))
        return samples

    @staticmethod
    def _load_protein(path: Path) -> List[ProteinSample]:
        with open(path, "r") as f:
            data = json.load(f)
        samples = []
        for entry in data["rows"]:
            row = entry["row"]
            samples.append(ProteinSample(
                row_idx=entry["row_idx"],
                protein_id=row.get("protein_id", "") or "",
                protein_names=row.get("protein_names", "") or "",
                protein_function=row.get("protein_function", "") or "",
                organism=row.get("organism", "") or "",
                length=row.get("length", 0.0) or 0.0,
                subcellular_location=row.get("subcellular_location", "") or "",
                sequence=row.get("sequence", "") or "",
                go_ids=row.get("go_ids") or [],
                go_bp=row.get("go_bp") or [],
                go_mf=row.get("go_mf") or [],
                go_cc=row.get("go_cc") or [],
                interpro_ids=row.get("interpro_ids") or [],
                interpro_formatted=row.get("interpro_formatted", "") or "",
                ppi_formatted=row.get("ppi_formatted", "") or "",
                go_pred=row.get("go_pred", "") or "",
                reasoning=row.get("reasoning", "") or "",
                final_answer=row.get("final_answer", "") or "",
                go_pred_leaf=row.get("go_pred_leaf", "") or "",
                go_bp_leaf=row.get("go_bp_leaf", "") or "",
                go_mf_leaf=row.get("go_mf_leaf", "") or "",
                go_cc_leaf=row.get("go_cc_leaf", "") or "",
                interaction_partners=row.get("interaction_partners") or [],
                interpro_location=row.get("interpro_location", "") or "",
                structure_path=row.get("structure_path") or "",
                string_id=row.get("string_id") or "",
            ))
        return samples

    @staticmethod
    def _load_catalogue(path: Path) -> List[CatalogueSample]:
        with open(path, "r") as f:
            data = json.load(f)
        samples = []
        for entry in data["rows"]:
            row = entry["row"]
            generation = row.get("generation", "") or ""
            steps, structured = _extract_think_steps(generation)
            samples.append(CatalogueSample(
                row_idx=entry["row_idx"],
                protein_id=row.get("protein_id", "") or "",
                sequence=row.get("protein", "") or "",
                organism=row.get("organism", "") or "",
                think_steps=steps,
                structured_answer=structured,
                raw_generation=generation,
            ))
        return samples

    @staticmethod
    def _load_catalogue_bridge(path: Path) -> List[ProteinSample]:
        """Load ``protein_catalogue_bridge.json`` as bridge-only ProteinSamples.

        The bridge file is NOT added to any training pool (the file's own
        description makes that explicit). It is loaded purely so that
        ``get_protein_by_uniprot`` resolves catalogue protein IDs for lab
        tool calls.
        """
        with open(path, "r") as f:
            data = json.load(f)
        samples = []
        for entry in data.get("rows", []):
            row = entry.get("row", {}) or {}
            samples.append(ProteinSample(
                row_idx=entry.get("row_idx", 0),
                protein_id=row.get("protein_id", "") or "",
                protein_names=row.get("protein_names", "") or "",
                protein_function=row.get("protein_function", "") or "",
                organism=row.get("organism", "") or "",
                length=float(row.get("length", 0.0) or 0.0),
                subcellular_location=row.get("subcellular_location", "") or "",
                sequence=row.get("sequence", "") or "",
                go_ids=row.get("go_ids") or [],
                go_bp=row.get("go_bp") or [],
                go_mf=row.get("go_mf") or [],
                go_cc=row.get("go_cc") or [],
                interpro_ids=row.get("interpro_ids") or [],
                interpro_formatted=row.get("interpro_formatted", "") or "",
                ppi_formatted=row.get("ppi_formatted", "") or "",
                reasoning=row.get("reasoning", "") or "",
                final_answer=row.get("final_answer", "") or "",
                go_pred_leaf=row.get("go_pred_leaf", "") or "",
                go_bp_leaf=row.get("go_bp_leaf", "") or "",
                go_mf_leaf=row.get("go_mf_leaf", "") or "",
                go_cc_leaf=row.get("go_cc_leaf", "") or "",
                interaction_partners=row.get("interaction_partners") or [],
            ))
        return samples

    @staticmethod
    def _load_diagnosis(path: Path) -> List[DiagnosisSample]:
        with open(path, "r") as f:
            data = json.load(f)
        samples = []
        for entry in data.get("rows", []):
            row = entry.get("row", {}) or {}
            raw_reasoning = row.get("gptoss120b_reasoning", "") or ""
            samples.append(DiagnosisSample(
                row_idx=entry.get("row_idx", 0),
                case_id=row.get("case_id", f"case_{entry.get('row_idx', 0):04d}") or "",
                description=row.get("PostDescription", "") or "",
                differentials=_split_differentials(row.get("DifferentialDiagnosisList", "") or ""),
                final_diagnosis=row.get("FinalDiagnosis", "") or "",
                reasoning_steps=_extract_diagnosis_steps(raw_reasoning),
                raw_reasoning=raw_reasoning,
            ))
        return samples

    @staticmethod
    def _load_perturbation(path: Path) -> List[PerturbationSample]:
        """Load perturbation pairs and derive (query_gene, target_gene) from each question."""
        with open(path, "r") as f:
            data = json.load(f)
        pair_re = re.compile(
            r"knocking\s+down\s+([A-Za-z0-9_\-]+).*?impact\s+the\s+expression\s+of\s+([A-Za-z0-9_\-]+)\s+in\s+([A-Za-z0-9_\-]+)\s*cells",
            re.IGNORECASE | re.DOTALL,
        )
        samples = []
        for entry in data.get("rows", []):
            row = entry.get("row", {}) or {}
            question = row.get("input", "") or ""
            answer_text = (row.get("output", "") or "").strip().lower()
            answer_bool = answer_text.startswith("y")
            match = pair_re.search(question)
            if match:
                query_gene = match.group(1).upper()
                target_gene = match.group(2).upper()
                cell_line = match.group(3).lower()
            else:
                query_gene = ""
                target_gene = ""
                cell_line = ""
            row_idx = entry.get("row_idx", 0)
            pair_id = f"pert_{row_idx:04d}"
            samples.append(PerturbationSample(
                row_idx=row_idx,
                pair_id=pair_id,
                query_gene=query_gene,
                target_gene=target_gene,
                cell_line=cell_line,
                question=question.strip(),
                answer=answer_bool,
            ))
        return samples

    @staticmethod
    def _load_hetionet(path: Path) -> List[LigandSample]:
        with open(path, "r") as f:
            data = json.load(f)
        samples = []
        for entry in data.get("rows", []):
            row = entry.get("row", {}) or {}
            gold_target = row.get("target", "") or ""
            is_smiles = gold_target.strip().startswith("[mol]")
            prompt = row.get("prompt", "") or ""
            go_neighbors_text = ""
            m = re.search(r"One-hop neighbors:\s*(.+?)\s*Answer:", prompt, re.DOTALL)
            if m:
                go_neighbors_text = m.group(1).strip()
            samples.append(LigandSample(
                row_idx=entry.get("row_idx", 0),
                gene=(row.get("gene", "") or "").upper(),
                go_neighbors_text=go_neighbors_text,
                gold_target=gold_target,
                gold_is_smiles=is_smiles,
                prompt=prompt,
            ))
        return samples

    @staticmethod
    def _load_top1000(path: Path) -> List[DrugRecord]:
        with open(path, "r") as f:
            data = json.load(f)
        records = []
        for row in data:
            try:
                records.append(DrugRecord(
                    smiles=row.get("SMILES", "") or "",
                    pic50=float(row.get("pIC50", 0.0) or 0.0),
                    num_atoms=int(row.get("num_atoms", 0) or 0),
                    logp=float(row.get("logP", 0.0) or 0.0),
                    drug_score=float(row.get("drug_score", 0.0) or 0.0),
                ))
            except Exception:
                continue
        return records

    @staticmethod
    def _load_kegg(path: Path, offset: int = 0) -> List[KeggSample]:
        """Load a KEGG pathway-networked reasoning file.

        Each row embeds a pathway graph string (``Network Definition of the
        pathway: ...``) and a semicolon-delimited gene list inside the
        ``question`` field. We pre-parse both to feed the grader.
        """
        with open(path, "r") as f:
            data = json.load(f)
        samples: List[KeggSample] = []
        for entry in data.get("rows", []):
            row = entry.get("row", {}) or {}
            row_idx = entry.get("row_idx", 0) + offset
            question = row.get("question", "") or ""
            pathway_graph = _extract_pathway_graph(question)
            genes = _extract_pathway_gene_symbols(question)
            raw_reasoning = row.get("reasoning", "") or ""
            samples.append(KeggSample(
                row_idx=row_idx,
                case_id=f"kegg_{row_idx:04d}",
                question=question,
                pathway_graph=pathway_graph,
                genes_in_pathway=genes,
                answer=(row.get("answer", "") or "").strip(),
                reasoning_steps=_extract_diagnosis_steps(raw_reasoning),
                raw_reasoning=raw_reasoning,
                reference_sequence=row.get("reference_sequence", "") or "",
                variant_sequence=row.get("variant_sequence", "") or "",
            ))
        return samples

    @staticmethod
    def _load_pert_variant(path: Path, variant: str) -> List[PerturbationDirSample]:
        """Load one of the four directional / GSE CRISPRi QA files."""
        with open(path, "r") as f:
            data = json.load(f)

        binary_pair_re = re.compile(
            r"knocking\s+down\s+([A-Za-z0-9_\-]+).*?(?:impact|increase or decrease)\s+the\s+expression\s+of\s+([A-Za-z0-9_\-]+)\s+in\s+([A-Za-z0-9_\-]+)\s*cells",
            re.IGNORECASE | re.DOTALL,
        )
        gse_pair_re = re.compile(
            r"(?:similar effects when (?:being )?knocked down as|respond similarly when being knocked down as)\s+([A-Za-z0-9_\-]+)\s+in\s+([A-Za-z0-9_\-]+)\s*cells",
            re.IGNORECASE | re.DOTALL,
        )

        samples: List[PerturbationDirSample] = []
        for entry in data.get("rows", []):
            row = entry.get("row", {}) or {}
            row_idx = entry.get("row_idx", 0)
            question = (row.get("input", "") or "").strip()
            answer = (row.get("output", "") or "").strip()

            query_gene = ""
            target_gene = ""
            cell_line = ""
            m = binary_pair_re.search(question)
            if m:
                query_gene = m.group(1).upper()
                target_gene = m.group(2).upper()
                cell_line = m.group(3).lower()
            else:
                g = gse_pair_re.search(question)
                if g:
                    query_gene = g.group(1).upper()
                    cell_line = g.group(2).lower()

            pair_id = f"{variant}_{row_idx:04d}"
            samples.append(PerturbationDirSample(
                row_idx=row_idx,
                pair_id=pair_id,
                query_gene=query_gene,
                target_gene=target_gene,
                cell_line=cell_line,
                question=question,
                answer=answer,
                variant=variant,
            ))
        return samples

    # -- ID-based access (GRPO same-prompt replay) -------------------------

    def get_dna_sample_by_id(self, task_id: str) -> DNASample:
        if task_id not in self._dna_by_id:
            raise KeyError(f"Unknown DNA task_id: {task_id}")
        return self._dna_by_id[task_id]

    def get_protein_sample_by_id(self, task_id: str) -> ProteinSample:
        if task_id not in self._protein_by_id:
            raise KeyError(f"Unknown protein task_id: {task_id}")
        return self._protein_by_id[task_id]

    def get_catalogue_sample_by_id(self, task_id: str) -> CatalogueSample:
        if task_id not in self._catalogue_by_id:
            raise KeyError(f"Unknown catalogue task_id: {task_id}")
        return self._catalogue_by_id[task_id]

    def get_protein_by_uniprot(self, protein_id: str) -> Optional[ProteinSample]:
        """Resolve a UniProt ID to a ``ProteinSample``.

        Priority: main protein pool -> catalogue bridge table (loaded from
        ``protein_catalogue_bridge.json``). This lets catalogue-sourced
        protein IDs resolve during lab tool calls without polluting the
        training pool.
        """
        sample = self._protein_by_uniprot.get(protein_id)
        if sample is not None:
            return sample
        return self._bridge_by_uniprot.get(protein_id)

    def get_catalogue_by_uniprot(self, protein_id: str) -> Optional[CatalogueSample]:
        return self._catalogue_by_uniprot.get(protein_id)

    def get_diagnosis_sample_by_id(self, task_id: str) -> DiagnosisSample:
        if task_id not in self._diagnosis_by_id:
            raise KeyError(f"Unknown diagnosis task_id: {task_id}")
        return self._diagnosis_by_id[task_id]

    def get_ligand_sample_by_id(self, task_id: str) -> LigandSample:
        if task_id not in self._ligand_by_id:
            raise KeyError(f"Unknown ligand task_id: {task_id}")
        return self._ligand_by_id[task_id]

    def get_perturbation_sample_by_id(self, pair_id: str) -> PerturbationSample:
        if pair_id not in self._perturbation_by_id:
            raise KeyError(f"Unknown perturbation pair_id: {pair_id}")
        return self._perturbation_by_id[pair_id]

    def get_kegg_sample_by_id(self, task_id: str) -> KeggSample:
        if task_id not in self._kegg_by_id:
            raise KeyError(f"Unknown KEGG task_id: {task_id}")
        return self._kegg_by_id[task_id]

    def get_pert_direction_sample_by_id(self, pair_id: str) -> PerturbationDirSample:
        if pair_id not in self._pert_dir_by_id:
            raise KeyError(f"Unknown perturbation-direction pair_id: {pair_id}")
        return self._pert_dir_by_id[pair_id]

    # -- Random sampling ---------------------------------------------------

    def get_random_dna_sample(self, rng: Optional[random.Random] = None) -> Tuple[str, DNASample]:
        pool = self._dna_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"dna_{sample.row_idx:03d}", sample

    def get_random_protein_sample(self, rng: Optional[random.Random] = None) -> Tuple[str, ProteinSample]:
        pool = self._protein_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"protein_{sample.row_idx:03d}", sample

    def get_random_catalogue_sample(self, rng: Optional[random.Random] = None) -> Tuple[str, CatalogueSample]:
        if not self._catalogue_samples:
            raise RuntimeError("Catalogue dataset not available")
        pool = self._catalogue_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"catalogue_{sample.row_idx:03d}", sample

    def get_random_diagnosis_sample(self, rng: Optional[random.Random] = None) -> Tuple[str, DiagnosisSample]:
        if not self._diagnosis_samples:
            raise RuntimeError("Diagnosis dataset not available")
        pool = self._diagnosis_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"diagnosis_{sample.row_idx:03d}", sample

    def get_random_ligand_sample(self, rng: Optional[random.Random] = None) -> Tuple[str, LigandSample]:
        if not self._ligand_samples:
            raise RuntimeError("Hetionet ligand dataset not available")
        pool = self._ligand_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"ligand_{sample.row_idx:03d}", sample

    def get_random_kegg_sample(self, rng: Optional[random.Random] = None) -> Tuple[str, KeggSample]:
        if not self._kegg_samples:
            raise RuntimeError("KEGG reasoning dataset not available")
        pool = self._kegg_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"kegg_{sample.row_idx:04d}", sample

    def get_perturbation_direction_batch(
        self,
        batch_id: str,
        batch_size: int = 10,
    ) -> List[PerturbationDirSample]:
        """Deterministic directional CRISPRi batch keyed by ``batch_id``.

        Always drawn from the ``pert_dir`` variant so the answer space is a
        clean 3-class problem (``Increase`` / ``Decrease`` / ``Unknown``).
        Hashing matches :py:meth:`get_perturbation_batch`.
        """
        pool = self._pert_variants.get("pert_dir", [])
        if not pool:
            return []
        n = len(pool)
        seed = int(hashlib.sha256(batch_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        indices = rng.sample(range(n), min(batch_size, n))
        return [pool[i] for i in sorted(indices)]

    def get_perturbation_benchmark_batch(
        self,
        batch_id: str,
        per_variant: int = 2,
    ) -> List[PerturbationDirSample]:
        """Deterministic umbrella batch: ``per_variant`` pairs from each of 4 variants."""
        seed_root = int(hashlib.sha256(batch_id.encode()).hexdigest(), 16) % (2**32)
        out: List[PerturbationDirSample] = []
        for i, variant in enumerate(["pert_dir", "pert_de", "gse_pert", "gse_gene"]):
            pool = self._pert_variants.get(variant, [])
            if not pool:
                continue
            rng = random.Random(seed_root + i)
            indices = rng.sample(range(len(pool)), min(per_variant, len(pool)))
            out.extend(pool[j] for j in sorted(indices))
        return out

    def get_perturbation_batch(
        self,
        batch_id: str,
        batch_size: int = 10,
    ) -> List[PerturbationSample]:
        """Return a deterministic batch of perturbation pairs for a given batch_id.

        Used by the ``perturbation_qa`` task. The mapping is pure (hash of
        batch_id -> indices) so the same ``batch_id`` always returns the
        same batch — GRPO-compatible.
        """
        if not self._perturbation_samples:
            return []
        n = len(self._perturbation_samples)
        seed = int(hashlib.sha256(batch_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        indices = rng.sample(range(n), min(batch_size, n))
        return [self._perturbation_samples[i] for i in sorted(indices)]

    # -- Batch access (GRPO iteration) -------------------------------------

    def get_all_dna_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"dna_{s.row_idx:03d}" for s in self._dna_samples[self.EPISODE_SPLIT:]]
        return [f"dna_{s.row_idx:03d}" for s in self._dna_samples[:self.EPISODE_SPLIT]]

    def get_all_protein_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"protein_{s.row_idx:03d}" for s in self._protein_samples[self.EPISODE_SPLIT:]]
        return [f"protein_{s.row_idx:03d}" for s in self._protein_samples[:self.EPISODE_SPLIT]]

    def get_all_catalogue_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"catalogue_{s.row_idx:03d}" for s in self._catalogue_samples[self.EPISODE_SPLIT:]]
        return [f"catalogue_{s.row_idx:03d}" for s in self._catalogue_samples[:self.EPISODE_SPLIT]]

    def get_all_diagnosis_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"diagnosis_{s.row_idx:03d}" for s in self._diagnosis_samples[self.EPISODE_SPLIT:]]
        return [f"diagnosis_{s.row_idx:03d}" for s in self._diagnosis_samples[:self.EPISODE_SPLIT]]

    def get_all_ligand_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"ligand_{s.row_idx:03d}" for s in self._ligand_samples[self.EPISODE_SPLIT:]]
        return [f"ligand_{s.row_idx:03d}" for s in self._ligand_samples[:self.EPISODE_SPLIT]]

    def get_all_perturbation_batch_ids(self, baseline_only: bool = False, n_batches: int = 40) -> List[str]:
        """Deterministic list of perturbation batch IDs for episode iteration."""
        if baseline_only:
            return [f"pertbatch_{i:03d}" for i in range(n_batches, n_batches + 10)]
        return [f"pertbatch_{i:03d}" for i in range(n_batches)]

    def get_all_kegg_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"kegg_{s.row_idx:04d}" for s in self._kegg_samples[self.EPISODE_SPLIT:]]
        return [f"kegg_{s.row_idx:04d}" for s in self._kegg_samples[:self.EPISODE_SPLIT]]

    def get_all_pert_direction_batch_ids(self, baseline_only: bool = False, n_batches: int = 30) -> List[str]:
        if baseline_only:
            return [f"pertdir_{i:03d}" for i in range(n_batches, n_batches + 10)]
        return [f"pertdir_{i:03d}" for i in range(n_batches)]

    def get_all_pert_benchmark_batch_ids(self, baseline_only: bool = False, n_batches: int = 20) -> List[str]:
        if baseline_only:
            return [f"pertbench_{i:03d}" for i in range(n_batches, n_batches + 10)]
        return [f"pertbench_{i:03d}" for i in range(n_batches)]

    def get_all_sample_ids(self, task_type: str, baseline_only: bool = False) -> List[str]:
        # Ordered by canonical narrative (Scene 1 → Scene 5).
        # Scene 1 + target_discovery_lab — DNA variant pool
        if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking", "target_discovery_lab"):
            return self.get_all_dna_ids(baseline_only)
        # Scene 2 + protein_hypothesis_lab — protein pool
        elif task_type in ("protein_function", "protein_hypothesis_lab"):
            return self.get_all_protein_ids(baseline_only)
        # Scene 3 — systems biology batches
        elif task_type == "kegg_pathway_reasoning":
            return self.get_all_kegg_ids(baseline_only)
        elif task_type == "perturbation_qa":
            return self.get_all_perturbation_batch_ids(baseline_only)
        elif task_type == "perturbation_direction_qa":
            return self.get_all_pert_direction_batch_ids(baseline_only)
        elif task_type == "perturbation_benchmark":
            return self.get_all_pert_benchmark_batch_ids(baseline_only)
        # Scene 4 + clinical_diagnosis_lab — radiology cases
        elif task_type in ("clinical_diagnosis", "clinical_diagnosis_lab"):
            return self.get_all_diagnosis_ids(baseline_only)
        # Scene 5 — remaining labs
        elif task_type == "ligand_design":
            return self.get_all_ligand_ids(baseline_only)
        elif task_type == "curriculum_self_play":
            return self.get_all_catalogue_ids(baseline_only)
        raise ValueError(f"Unknown task_type: {task_type}")

    # -- Distractor selection for Task 4 -----------------------------------

    def get_distractors(self, task_id: str, n: int = 3) -> List[str]:
        """Return n distractor disease names for a given DNA task_id."""
        gold_sample = self.get_dna_sample_by_id(task_id)
        gold_disease = gold_sample.answer.lower().strip()

        candidates = [d for d in self._all_disease_answers if d != gold_disease]

        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        rng.shuffle(candidates)

        return candidates[:n]

    def get_candidate_diseases(self, task_id: str) -> List[str]:
        gold_sample = self.get_dna_sample_by_id(task_id)
        gold_disease = gold_sample.answer.lower().strip()
        distractors = self.get_distractors(task_id, n=3)

        candidates = [gold_disease] + distractors
        seed = int(hashlib.sha256((task_id + "_shuffle").encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        return candidates

    # -- Tool responses (used by lab-mode environment) ---------------------

    def tool_response(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call to a deterministic response.

        All responses are pure functions of (tool_name, args) and the
        underlying datasets, which satisfies GRPO's same-prompt replay
        requirement. Unknown tools / arguments return an ``error`` payload.
        """
        args = args or {}
        try:
            if tool_name == "get_interpro":
                return self._tool_get_interpro(args)
            if tool_name == "get_ppi":
                return self._tool_get_ppi(args)
            if tool_name == "get_go":
                return self._tool_get_go(args)
            if tool_name == "get_sequence":
                return self._tool_get_sequence(args)
            if tool_name == "get_pathway":
                return self._tool_get_pathway(args)
            if tool_name == "search_catalogue":
                return self._tool_search_catalogue(args)
            if tool_name == "get_subcellular_location":
                return self._tool_get_location(args)
            if tool_name == "get_drug_properties":
                return self._tool_get_drug_properties(args)
            if tool_name == "get_candidate_ligands":
                return self._tool_get_candidate_ligands(args)
            if tool_name == "get_perturbation_pair":
                return self._tool_get_perturbation_pair(args)
            if tool_name == "get_structure":
                return self._tool_get_structure(args)
            return {"error": f"unknown tool: {tool_name}"}
        except Exception as exc:  # pragma: no cover — defensive
            return {"error": f"tool raised exception: {exc}"}

    @staticmethod
    def _cap(text: str, cap: int) -> str:
        if not text:
            return ""
        return text if len(text) <= cap else text[: cap - 3] + "..."

    def _tool_get_interpro(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = args.get("protein_id", "")
        sample = self.get_protein_by_uniprot(pid)
        if sample is None:
            # Fallback to catalogue
            cat = self.get_catalogue_by_uniprot(pid)
            if cat is None:
                return {"error": f"unknown protein_id: {pid}"}
            return {"protein_id": pid, "interpro_formatted": self._cap(cat.raw_generation[:800], self.NOTEBOOK_CHAR_CAP)}
        return {
            "protein_id": pid,
            "interpro_ids": sample.interpro_ids,
            "interpro_formatted": self._cap(sample.interpro_formatted, self.NOTEBOOK_CHAR_CAP),
        }

    def _tool_get_ppi(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = args.get("protein_id", "")
        sample = self.get_protein_by_uniprot(pid)
        if sample is None:
            return {"error": f"unknown protein_id: {pid}"}
        return {
            "protein_id": pid,
            "interaction_partners": sample.interaction_partners,
            "ppi_formatted": self._cap(sample.ppi_formatted, self.NOTEBOOK_CHAR_CAP),
        }

    def _tool_get_go(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = args.get("protein_id", "")
        branch = (args.get("branch") or "all").lower()
        sample = self.get_protein_by_uniprot(pid)
        if sample is None:
            return {"error": f"unknown protein_id: {pid}"}
        if branch == "bp":
            return {"protein_id": pid, "branch": "bp", "go": self._cap(sample.go_bp_leaf or "\n".join(sample.go_bp), self.NOTEBOOK_CHAR_CAP)}
        if branch == "mf":
            return {"protein_id": pid, "branch": "mf", "go": self._cap(sample.go_mf_leaf or "\n".join(sample.go_mf), self.NOTEBOOK_CHAR_CAP)}
        if branch == "cc":
            return {"protein_id": pid, "branch": "cc", "go": self._cap(sample.go_cc_leaf or "\n".join(sample.go_cc), self.NOTEBOOK_CHAR_CAP)}
        if branch == "leaf":
            combined = "\n".join(x for x in [sample.go_bp_leaf, sample.go_mf_leaf, sample.go_cc_leaf] if x)
            return {"protein_id": pid, "branch": "leaf", "go": self._cap(combined, self.NOTEBOOK_CHAR_CAP)}
        # default: all
        return {
            "protein_id": pid,
            "branch": "all",
            "go_ids": sample.go_ids[:20],
            "go_bp": sample.go_bp[:10],
            "go_mf": sample.go_mf[:10],
            "go_cc": sample.go_cc[:10],
        }

    def _tool_get_sequence(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = args.get("protein_id", "")
        window = args.get("window")
        sample = self.get_protein_by_uniprot(pid)
        if sample is None:
            cat = self.get_catalogue_by_uniprot(pid)
            if cat is None:
                return {"error": f"unknown protein_id: {pid}"}
            seq = cat.sequence
            organism = cat.organism
        else:
            seq = sample.sequence
            organism = sample.organism
        if isinstance(window, (list, tuple)) and len(window) == 2:
            try:
                start, end = int(window[0]), int(window[1])
                seq = seq[max(0, start - 1): max(0, end)]
            except Exception:
                pass
        return {
            "protein_id": pid,
            "organism": organism,
            "sequence": self._cap(seq, self.NOTEBOOK_CHAR_CAP),
            "length": len(seq),
        }

    def _tool_get_location(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pid = args.get("protein_id", "")
        sample = self.get_protein_by_uniprot(pid)
        if sample is None:
            return {"error": f"unknown protein_id: {pid}"}
        return {
            "protein_id": pid,
            "subcellular_location": self._cap(sample.subcellular_location, self.NOTEBOOK_CHAR_CAP),
        }

    def _tool_get_pathway(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return pathway context (gene list + network) for a DNA task_id or gene symbol."""
        task_id = args.get("task_id", "")
        gene = (args.get("gene") or "").upper().strip()

        if task_id and task_id in self._dna_by_id:
            sample = self._dna_by_id[task_id]
            return {
                "task_id": task_id,
                "pathway_excerpt": self._cap(
                    _extract_pathway_section(sample.question),
                    self.NOTEBOOK_CHAR_CAP,
                ),
                "genes": _extract_pathway_genes(sample.question),
            }

        if gene:
            # Search DNA samples whose gene list contains ``gene``
            hits: List[str] = []
            for sid, sample in self._dna_by_id.items():
                if gene in _extract_pathway_genes(sample.question):
                    hits.append(sid)
                if len(hits) >= 5:
                    break
            return {"gene": gene, "matches": hits}

        return {"error": "tool requires either 'task_id' or 'gene'"}

    def _tool_search_catalogue(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return up to 5 catalogue entries whose generation mentions ``keyword``."""
        keyword = (args.get("keyword") or "").lower().strip()
        if not keyword:
            return {"error": "tool requires 'keyword'"}
        hits: List[Dict[str, Any]] = []
        for sample in self._catalogue_samples:
            if keyword in sample.raw_generation.lower():
                hits.append({
                    "catalogue_id": f"catalogue_{sample.row_idx:03d}",
                    "protein_id": sample.protein_id,
                    "organism": sample.organism,
                    "preview": self._cap(sample.structured_answer[:300], self.NOTEBOOK_CHAR_CAP),
                })
            if len(hits) >= 5:
                break
        return {"keyword": keyword, "matches": hits}

    def _tool_get_drug_properties(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Look up a SMILES string in the top-1000 pool or compute fallback stats."""
        smiles = (args.get("smiles") or "").strip()
        if not smiles:
            return {"error": "tool requires 'smiles'"}
        record = self._top1000_by_smiles.get(smiles)
        if record is not None:
            return {
                "smiles": record.smiles,
                "pIC50": round(record.pic50, 4),
                "num_atoms": record.num_atoms,
                "logP": round(record.logp, 4),
                "drug_score": round(record.drug_score, 4),
                "in_catalogue": True,
            }
        # Deterministic fallback stats (no rdkit).
        alpha = sum(1 for c in smiles if c.isalpha())
        return {
            "smiles": self._cap(smiles, self.NOTEBOOK_CHAR_CAP),
            "pIC50": None,
            "num_atoms": alpha,
            "logP": 0.0,
            "drug_score": 0.0,
            "in_catalogue": False,
        }

    def _tool_get_candidate_ligands(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return up to k deterministic ligand candidates for a gene symbol.

        Candidates are drawn from the hetionet ligand table joined with the
        top-1000 molecule pool, plus a handful of high drug_score fillers.
        """
        gene = (args.get("gene") or "").upper().strip()
        try:
            k = int(args.get("k", 5) or 5)
        except Exception:
            k = 5
        k = max(1, min(10, k))
        if not gene:
            return {"error": "tool requires 'gene'"}

        candidates: List[Dict[str, Any]] = []
        seen_smiles: set = set()
        for lig in self._ligands_by_gene.get(gene, []):
            if lig.gold_is_smiles:
                smiles = lig.gold_target
            else:
                smiles = lig.gold_target  # named drug
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            record = self._top1000_by_smiles.get(smiles)
            candidates.append({
                "smiles": smiles if lig.gold_is_smiles else "",
                "name": "" if lig.gold_is_smiles else lig.gold_target,
                "pIC50": round(record.pic50, 3) if record else None,
                "drug_score": round(record.drug_score, 3) if record else None,
                "source": "hetionet",
            })
            if len(candidates) >= k:
                break

        # Fill with top-ranked molecules from the catalogue if not enough.
        if len(candidates) < k:
            pool = sorted(self._top1000, key=lambda d: d.drug_score, reverse=True)
            for d in pool:
                if d.smiles in seen_smiles:
                    continue
                seen_smiles.add(d.smiles)
                candidates.append({
                    "smiles": d.smiles,
                    "name": "",
                    "pIC50": round(d.pic50, 3),
                    "drug_score": round(d.drug_score, 3),
                    "source": "top1000",
                })
                if len(candidates) >= k:
                    break

        return {"gene": gene, "candidates": candidates[:k]}

    def _tool_get_structure(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return the AlphaFold structure filename + deterministic signature.

        New in v3: proteins sourced from ``Protien_data_2.json`` /
        ``Protien_sft_reasoning_2.json`` carry a ``structure_path`` field
        (e.g. ``AF-Q13148-F1-model_v6.pdb``). This tool lets the agent
        reference the structure during a lab episode without requiring a
        heavyweight structure parser — the ``signature`` is a SHA256-based
        16-char hash so the agent has a stable token to quote in its
        reasoning chain.
        """
        pid = (args.get("protein_id") or "").strip()
        if not pid:
            return {"error": "tool requires 'protein_id'"}
        sample = self.get_protein_by_uniprot(pid)
        if sample is None or not sample.structure_path:
            return {
                "protein_id": pid,
                "error": "not_in_catalogue",
                "source": "AlphaFold",
            }
        signature = hashlib.sha256(sample.structure_path.encode()).hexdigest()[:16]
        return {
            "protein_id": pid,
            "structure_path": sample.structure_path,
            "signature": signature,
            "source": "AlphaFold",
        }

    def _tool_get_perturbation_pair(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup a single CRISPRi pair by gene symbols.

        The lab-mode environment guards this tool so it is NEVER exposed
        during ``perturbation_qa`` episodes (that would leak labels).
        """
        query = (args.get("query_gene") or "").upper().strip()
        target = (args.get("target_gene") or "").upper().strip()
        if not query or not target:
            return {"error": "tool requires 'query_gene' and 'target_gene'"}
        for s in self._perturbation_samples:
            if s.query_gene == query and s.target_gene == target:
                return {
                    "query_gene": query,
                    "target_gene": target,
                    "cell_line": s.cell_line,
                    "answer": "yes" if s.answer else "no",
                }
        return {
            "query_gene": query,
            "target_gene": target,
            "answer": "unknown",
        }

    # -- Properties --------------------------------------------------------

    @property
    def all_disease_answers(self) -> List[str]:
        return list(self._all_disease_answers)

    @property
    def dna_count(self) -> int:
        return len(self._dna_samples)

    @property
    def protein_count(self) -> int:
        return len(self._protein_samples)

    @property
    def catalogue_count(self) -> int:
        return len(self._catalogue_samples)

    @property
    def diagnosis_count(self) -> int:
        return len(self._diagnosis_samples)

    @property
    def perturbation_count(self) -> int:
        return len(self._perturbation_samples)

    @property
    def ligand_count(self) -> int:
        return len(self._ligand_samples)

    @property
    def top1000_count(self) -> int:
        return len(self._top1000)

    @property
    def top1000(self) -> List[DrugRecord]:
        return list(self._top1000)

    @property
    def top1000_by_smiles(self) -> Dict[str, DrugRecord]:
        return dict(self._top1000_by_smiles)

    @property
    def kegg_count(self) -> int:
        return len(self._kegg_samples)

    @property
    def pert_direction_count(self) -> int:
        return sum(len(v) for v in self._pert_variants.values())

    def pert_variant_count(self, variant: str) -> int:
        return len(self._pert_variants.get(variant, []))


# =========================================================================
# Shared helpers (also used by environment & graders)
# =========================================================================


def _extract_pathway_section(question: str) -> str:
    """Extract the 'Network Definition' + 'Genes in the pathway' section."""
    network = re.search(r"Network Definition[^\n]*", question)
    genes = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    parts = []
    if network:
        parts.append(network.group(0))
    if genes:
        parts.append("Genes: " + genes.group(1).strip())
    return "\n".join(parts)


def _extract_pathway_genes(question: str) -> List[str]:
    """Extract gene symbols from the question's pathway gene list."""
    genes: List[str] = []
    section = re.search(r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context)", question, re.DOTALL)
    if section:
        for match in re.finditer(r"(\w+)\s*;", section.group(1)):
            genes.append(match.group(1))
    return genes


def _extract_pathway_graph(question: str) -> str:
    """Pull the raw KEGG declarative graph line (``TARDBP* -| CxI -> Q``)."""
    match = re.search(
        r"Network Definition of the pathway\s*:\s*(.+?)\n",
        question,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _extract_pathway_gene_symbols(question: str) -> List[str]:
    """Parse the 'Genes in the pathway:' block into a clean symbol list.

    The KEGG rows format the block as ``SYMBOL; Full Name | SYMBOL2; ...``.
    Only the leading token of each semicolon block is kept.
    """
    section = re.search(
        r"Genes in the pathway:\s*(.+?)(?:\n\n|Given this context|$)",
        question,
        re.DOTALL,
    )
    if not section:
        return []
    genes: List[str] = []
    for block in section.group(1).split("|"):
        block = block.strip()
        if not block:
            continue
        # Take the text before the first semicolon as the gene symbol.
        sym = block.split(";", 1)[0].strip()
        # Filter out nonsense tokens (common non-gene prefixes).
        if sym and re.fullmatch(r"[A-Za-z0-9\-_]+", sym):
            genes.append(sym.upper())
    return genes
