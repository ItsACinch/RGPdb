"""Knowledge triple loading from CSV/JSON files."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from ..config import RELATION_TO_BIN, relation_to_bin


@dataclass
class Triple:
    """A knowledge triple (head, relation, tail)."""

    head: str
    relation: str
    tail: str

    def __hash__(self):
        return hash((self.head, self.relation, self.tail))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return self.head == other.head and self.relation == other.relation and self.tail == other.tail


@dataclass
class MappedTriple:
    """A triple with relation mapped to RGDB angle bin."""

    head: str
    head_id: int
    relation: str
    angle_bin: int
    tail: str
    tail_id: int

    def as_tuple(self) -> Tuple[int, int, int]:
        """Return as (head_id, angle_bin, tail_id) tuple."""
        return (self.head_id, self.angle_bin, self.tail_id)


class TripleLoader:
    """Load and process knowledge triples."""

    def __init__(self):
        """Initialize the triple loader."""
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}

    def load(self, path: Union[str, Path]) -> List[Triple]:
        """
        Load triples from a file.

        Supports CSV and JSON formats.

        CSV format (with header):
            head,relation,tail
            machine_learning,is_a,artificial_intelligence
            ...

        JSON format:
            [
                {"head": "machine_learning", "relation": "is_a", "tail": "artificial_intelligence"},
                ...
            ]

        Args:
            path: Path to the triples file

        Returns:
            List of Triple objects
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()

        if extension == ".csv":
            return self._load_csv(path)
        elif extension == ".json":
            return self._load_json(path)
        else:
            # Try to detect format
            content = path.read_text()
            if content.strip().startswith("["):
                return self._load_json(path)
            else:
                return self._load_csv(path)

    def _load_csv(self, path: Path) -> List[Triple]:
        """Load triples from CSV file."""
        triples = []

        with open(path, "r", encoding="utf-8", newline="") as f:
            # Try to detect if there's a header
            sample = f.read(1024)
            f.seek(0)

            has_header = "head" in sample.lower() and "tail" in sample.lower()

            reader = csv.reader(f)

            if has_header:
                next(reader)  # Skip header

            for row in reader:
                if len(row) >= 3:
                    head, relation, tail = row[0].strip(), row[1].strip(), row[2].strip()
                    if head and relation and tail:
                        triples.append(Triple(head=head, relation=relation, tail=tail))

        return triples

    def _load_json(self, path: Path) -> List[Triple]:
        """Load triples from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        triples = []
        for item in data:
            if isinstance(item, dict):
                head = item.get("head", item.get("h", item.get("subject", "")))
                relation = item.get("relation", item.get("r", item.get("predicate", "")))
                tail = item.get("tail", item.get("t", item.get("object", "")))

                if head and relation and tail:
                    triples.append(Triple(head=str(head), relation=str(relation), tail=str(tail)))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                triples.append(Triple(head=str(item[0]), relation=str(item[1]), tail=str(item[2])))

        return triples

    def build_entity_vocab(self, triples: List[Triple]) -> Dict[str, int]:
        """
        Build entity vocabulary from triples.

        Args:
            triples: List of triples

        Returns:
            Dictionary mapping entity strings to IDs
        """
        entities: Set[str] = set()

        for triple in triples:
            entities.add(triple.head)
            entities.add(triple.tail)

        self.entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
        self.id_to_entity = {idx: entity for entity, idx in self.entity_to_id.items()}

        return self.entity_to_id

    def map_relations_to_bins(
        self, triples: List[Triple], entity_vocab: Optional[Dict[str, int]] = None
    ) -> List[MappedTriple]:
        """
        Convert triples to MappedTriples with RGDB angle bins.

        Args:
            triples: List of Triple objects
            entity_vocab: Optional entity vocabulary. If None, builds from triples.

        Returns:
            List of MappedTriple objects
        """
        if entity_vocab is None:
            entity_vocab = self.build_entity_vocab(triples)

        mapped = []
        for triple in triples:
            head_id = entity_vocab.get(triple.head)
            tail_id = entity_vocab.get(triple.tail)

            if head_id is None or tail_id is None:
                continue

            angle_bin = relation_to_bin(triple.relation)

            mapped.append(
                MappedTriple(
                    head=triple.head,
                    head_id=head_id,
                    relation=triple.relation,
                    angle_bin=angle_bin,
                    tail=triple.tail,
                    tail_id=tail_id,
                )
            )

        return mapped

    def get_relation_statistics(self, triples: List[Triple]) -> Dict[str, int]:
        """Get counts of each relation type."""
        counts: Dict[str, int] = {}
        for triple in triples:
            rel = triple.relation.lower()
            counts[rel] = counts.get(rel, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_angle_bin_statistics(self, mapped: List[MappedTriple]) -> Dict[int, int]:
        """Get counts of each angle bin."""
        counts: Dict[int, int] = {}
        for triple in mapped:
            counts[triple.angle_bin] = counts.get(triple.angle_bin, 0) + 1
        return dict(sorted(counts.items()))

    def split_train_test(
        self, triples: List[Triple], test_ratio: float = 0.1, seed: int = 42
    ) -> Tuple[List[Triple], List[Triple]]:
        """
        Split triples into train and test sets.

        Args:
            triples: List of triples to split
            test_ratio: Fraction of data to use for testing (0.0-1.0)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_triples, test_triples)

        Raises:
            ValueError: If test_ratio is not between 0 and 1
        """
        import random

        if not 0.0 <= test_ratio <= 1.0:
            raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

        # Use local Random instance to avoid affecting global state
        rng = random.Random(seed)
        shuffled = list(triples)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - test_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]


def save_vocab(vocab: Dict[str, int], path: Union[str, Path]) -> None:
    """Save entity vocabulary to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


def load_vocab(path: Union[str, Path]) -> Dict[str, int]:
    """Load entity vocabulary from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
