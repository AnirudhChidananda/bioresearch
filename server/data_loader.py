"""
Data loading and sampling for the Bioresearch Environment.

Handles the HuggingFace dataset JSON format (features + rows),
provides typed dataclass access, and supports GRPO-compatible
deterministic same-prompt replay via task_id lookups.
"""

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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


class DataLoader:
    """Loads and manages bioresearch datasets with GRPO-compatible access patterns."""

    EPISODE_SPLIT = 80  # first 80 samples for episodes, last 20 for baseline

    def __init__(self, data_dir: Optional[str] = None):
        data_path = Path(data_dir) if data_dir else _find_data_dir()

        self._dna_samples: List[DNASample] = self._load_dna(data_path / "DNA_reasoning.json")
        self._protein_samples: List[ProteinSample] = self._load_protein(data_path / "Protein_reasoning.json")

        self._dna_by_id: Dict[str, DNASample] = {
            f"dna_{s.row_idx:03d}": s for s in self._dna_samples
        }
        self._protein_by_id: Dict[str, ProteinSample] = {
            f"protein_{s.row_idx:03d}": s for s in self._protein_samples
        }

        self._all_disease_answers: List[str] = list({
            s.answer.lower().strip() for s in self._dna_samples
        })
        self._all_disease_answers.sort()

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
                protein_id=row.get("protein_id", ""),
                protein_names=row.get("protein_names", ""),
                protein_function=row.get("protein_function", ""),
                organism=row.get("organism", ""),
                length=row.get("length", 0.0),
                subcellular_location=row.get("subcellular_location", ""),
                sequence=row.get("sequence", ""),
                go_ids=row.get("go_ids", []) or [],
                go_bp=row.get("go_bp", []) or [],
                go_mf=row.get("go_mf", []) or [],
                go_cc=row.get("go_cc", []) or [],
                interpro_ids=row.get("interpro_ids", []) or [],
                interpro_formatted=row.get("interpro_formatted", ""),
                ppi_formatted=row.get("ppi_formatted", ""),
                go_pred=row.get("go_pred", ""),
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

    # -- Random sampling ---------------------------------------------------

    def get_random_dna_sample(self, rng: Optional[random.Random] = None) -> tuple:
        """Returns (task_id, DNASample) from the episode pool."""
        pool = self._dna_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"dna_{sample.row_idx:03d}", sample

    def get_random_protein_sample(self, rng: Optional[random.Random] = None) -> tuple:
        """Returns (task_id, ProteinSample) from the episode pool."""
        pool = self._protein_samples[:self.EPISODE_SPLIT]
        r = rng or random
        sample = r.choice(pool)
        return f"protein_{sample.row_idx:03d}", sample

    # -- Batch access (GRPO iteration) -------------------------------------

    def get_all_dna_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"dna_{s.row_idx:03d}" for s in self._dna_samples[self.EPISODE_SPLIT:]]
        return [f"dna_{s.row_idx:03d}" for s in self._dna_samples[:self.EPISODE_SPLIT]]

    def get_all_protein_ids(self, baseline_only: bool = False) -> List[str]:
        if baseline_only:
            return [f"protein_{s.row_idx:03d}" for s in self._protein_samples[self.EPISODE_SPLIT:]]
        return [f"protein_{s.row_idx:03d}" for s in self._protein_samples[:self.EPISODE_SPLIT]]

    def get_all_sample_ids(self, task_type: str, baseline_only: bool = False) -> List[str]:
        if task_type in ("dna_classification", "dna_reasoning", "evidence_ranking"):
            return self.get_all_dna_ids(baseline_only)
        elif task_type == "protein_function":
            return self.get_all_protein_ids(baseline_only)
        raise ValueError(f"Unknown task_type: {task_type}")

    # -- Distractor selection for Task 4 -----------------------------------

    def get_distractors(self, task_id: str, n: int = 3) -> List[str]:
        """
        Return n distractor disease names for a given DNA task_id.
        Selection is deterministic (hash-based) and avoids the gold answer.
        Prefers diseases from different pathway families.
        """
        gold_sample = self.get_dna_sample_by_id(task_id)
        gold_disease = gold_sample.answer.lower().strip()

        candidates = [d for d in self._all_disease_answers if d != gold_disease]

        seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        rng.shuffle(candidates)

        return candidates[:n]

    def get_candidate_diseases(self, task_id: str) -> List[str]:
        """
        Return 4 candidate diseases (1 gold + 3 distractors), shuffled deterministically.
        """
        gold_sample = self.get_dna_sample_by_id(task_id)
        gold_disease = gold_sample.answer.lower().strip()
        distractors = self.get_distractors(task_id, n=3)

        candidates = [gold_disease] + distractors
        seed = int(hashlib.sha256((task_id + "_shuffle").encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        return candidates

    @property
    def all_disease_answers(self) -> List[str]:
        return list(self._all_disease_answers)

    @property
    def dna_count(self) -> int:
        return len(self._dna_samples)

    @property
    def protein_count(self) -> int:
        return len(self._protein_samples)
