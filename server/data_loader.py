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

        self._dna_samples: List[DNASample] = self._load_dna(dna_path)
        self._protein_samples: List[ProteinSample] = self._load_protein(protein_path)
        self._catalogue_samples: List[CatalogueSample] = (
            self._load_catalogue(catalogue_path) if catalogue_path else []
        )

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
        return self._protein_by_uniprot.get(protein_id)

    def get_catalogue_by_uniprot(self, protein_id: str) -> Optional[CatalogueSample]:
        return self._catalogue_by_uniprot.get(protein_id)

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

    def get_all_sample_ids(self, task_type: str, baseline_only: bool = False) -> List[str]:
        if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking", "target_discovery_lab"):
            return self.get_all_dna_ids(baseline_only)
        elif task_type in ("protein_function", "protein_hypothesis_lab"):
            return self.get_all_protein_ids(baseline_only)
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
