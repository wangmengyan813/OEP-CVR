"""RDF/SPARQL class bindings, finite interpretation programs, and three-valued semantics."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef


ATTRIBUTES = ("discipline", "stage", "name", "unit", "twists")
SCHEMA = Namespace("urn:oep-cvr:")


class Truth(Enum):
    FALSE = 0
    UNKNOWN = 1
    TRUE = 2


@dataclass(frozen=True)
class Atom:
    op: str
    attribute: str
    value: str | int | None = None

    def __post_init__(self) -> None:
        if self.op not in {"set", "negate", "retain", "replace", "add", "remove"}:
            raise ValueError(f"Unsupported semantic operation: {self.op}")
        if self.attribute not in ATTRIBUTES:
            raise ValueError(f"Unsupported action attribute: {self.attribute}")
        if self.op in {"set", "replace", "add", "negate"} and self.value is None:
            raise ValueError("This operation requires a target value.")
        if self.op == "add" and (self.attribute != "twists" or not isinstance(self.value, int)):
            raise ValueError("Numeric addition is restricted to integer half-turn counts.")


@dataclass(frozen=True)
class Program:
    atoms: tuple[Atom, ...]


@dataclass(frozen=True)
class Bounds:
    lower: frozenset[str]
    upper: frozenset[str]
    fallback: bool = False
    reason: str = ""


class Ontology:
    """A finite OWL class graph with explicit attribute bindings and subclass closure.

    Additional OWL 2 consequences can be provided through pre-materialized RDF.
    """

    def __init__(self, records: list[dict], rdf_path: str | Path | None = None):
        if not records:
            raise ValueError("The ontology vocabulary must not be empty.")
        self.records = records
        self.ids = tuple(str(record["id"]) for record in records)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Ontology class identifiers must be unique.")
        self.index = {name: i for i, name in enumerate(self.ids)}
        self.graph = Graph()
        if rdf_path is not None:
            self.graph.parse(str(rdf_path))
        self.uris = {}
        for record in records:
            name = str(record["id"])
            node = URIRef(record.get("uri", str(SCHEMA) + "class/" + quote(name, safe="")))
            self.uris[name] = node
            self.graph.add((node, RDF.type, OWL.Class))
            for attribute, value in record.get("attributes", {}).items():
                if attribute not in ATTRIBUTES:
                    raise ValueError(f"Unknown ontology attribute: {attribute}")
                if value is not None:
                    if attribute == "twists" and (not isinstance(value, int) or isinstance(value, bool)):
                        raise ValueError("Twists must be integer half-turns.")
                    self.graph.add((node, SCHEMA[attribute], Literal(value)))
        for record in records:
            for parent in record.get("parents", []):
                if str(parent) not in self.index:
                    raise ValueError(f"Unknown parent class: {parent}")
                self.graph.add((self.uris[str(record["id"])], RDFS.subClassOf,
                                self.uris[str(parent)]))

    @classmethod
    def load(cls, filename: str | Path) -> "Ontology":
        filename = Path(filename)
        source = json.loads(filename.read_text(encoding="utf-8"))
        rdf = source.get("materialized_rdf")
        return cls(source["classes"], filename.parent / rdf if rdf else None)

    @lru_cache(maxsize=4096)
    def value(self, class_id: str, attribute: str) -> str | int | None:
        if class_id not in self.uris or attribute not in ATTRIBUTES:
            return None
        # The property path applies both asserted attributes and superclass bindings.
        query = ("SELECT DISTINCT ?v WHERE { "
                 f"<{self.uris[class_id]}> <{RDFS.subClassOf}>* ?ancestor . "
                 f"?ancestor <{SCHEMA[attribute]}> ?v . }}")
        values = {v.toPython() for row in self.graph.query(query) for v in [row[0]]}
        if len(values) != 1:
            return None
        result = next(iter(values))
        return int(result) if attribute == "twists" else str(result)

    def fallback(self, reason: str) -> Bounds:
        return Bounds(frozenset(), frozenset(self.ids), True, reason)

    def evaluate(self, target: str, reference: str, atom: Atom) -> Truth:
        current = self.value(target, atom.attribute)
        origin = self.value(reference, atom.attribute)
        if current is None:
            return Truth.UNKNOWN
        if atom.op == "set":
            condition = current == atom.value
        elif atom.op == "negate":
            condition = current != atom.value
        elif atom.op == "retain":
            if origin is None:
                return Truth.UNKNOWN
            condition = current == origin
        elif atom.op == "remove":
            if origin is None:
                return Truth.UNKNOWN
            condition = current != origin
        elif atom.op == "replace":
            if origin is None:
                return Truth.UNKNOWN
            condition = current == atom.value and origin != atom.value
        else:
            if origin is None or not isinstance(origin, int):
                return Truth.UNKNOWN
            condition = current == origin + atom.value
        return Truth.TRUE if condition else Truth.FALSE

    @lru_cache(maxsize=512)
    def bounds(self, reference_classes: frozenset[str], programs: tuple[Program, ...],
               max_interpretations: int = 16, max_conditions: int = 12) -> Bounds:
        if not reference_classes or not reference_classes.issubset(self.index):
            return self.fallback("reference-set mismatch")
        if not programs or len(programs) > max_interpretations:
            return self.fallback("invalid interpretation count")
        if any(not p.atoms or len(p.atoms) > max_conditions or invalid_program(p)
               for p in programs):
            return self.fallback("invalid or contradictory program")
        lower = set(self.ids)
        upper: set[str] = set()
        for origin in reference_classes:
            for program in programs:
                certain, possible = set(), set()
                for target in self.ids:
                    state = tuple(self.evaluate(target, origin, atom) for atom in program.atoms)
                    if all(item == Truth.TRUE for item in state):
                        certain.add(target)
                    if all(item != Truth.FALSE for item in state):
                        possible.add(target)
                lower.intersection_update(certain)
                upper.update(possible)
        if not upper:
            return self.fallback("empty possible-target bound")
        return Bounds(frozenset(lower), frozenset(upper))


def invalid_program(program: Program) -> bool:
    by_attribute: dict[str, list[Atom]] = {}
    for atom in program.atoms:
        by_attribute.setdefault(atom.attribute, []).append(atom)
    for group in by_attribute.values():
        equalities = [a.value for a in group if a.op == "set"]
        if len(set(equalities)) > 1:
            return True
        if any(a.op == "retain" for a in group) and any(
                a.op in {"replace", "remove"} for a in group):
            return True
        if equalities and any(a.op == "negate" and a.value == equalities[0] for a in group):
            return True
    return False


class Grammar:
    """Frozen, training-derived expression patterns mapped to finite programs."""

    def __init__(self, mappings: dict[str, Any], rules: list[dict] | None = None,
                 max_interpretations: int = 16, max_conditions: int = 12):
        self.mappings = mappings
        self.rules = [(re.compile(rule["pattern"], flags=re.IGNORECASE), rule["programs"])
                      for rule in (rules or [])]
        self.max_interpretations = max_interpretations
        self.max_conditions = max_conditions

    @classmethod
    def load(cls, filename: str | Path, max_interpretations: int = 16,
             max_conditions: int = 12) -> "Grammar":
        source = json.loads(Path(filename).read_text(encoding="utf-8"))
        return cls(source.get("mappings", {}), source.get("rules", []),
                   max_interpretations, max_conditions)

    def parse(self, modification: str) -> tuple[Program, ...]:
        raw_programs = self.mappings.get(modification)
        matches: dict[str, str] = {}
        if raw_programs is None:
            for pattern, programs in self.rules:
                match = pattern.fullmatch(modification.strip())
                if match:
                    raw_programs, matches = programs, match.groupdict()
                    break
        if raw_programs is None or len(raw_programs) > self.max_interpretations:
            return ()
        try:
            parsed: list[Program] = []
            for raw in raw_programs:
                if not 0 < len(raw) <= self.max_conditions:
                    return ()
                atoms = []
                for instruction in raw:
                    value = instruction.get("value")
                    if isinstance(value, str) and matches:
                        value = value.format_map(matches)
                    if instruction["attribute"] == "twists" and value is not None:
                        value = int(value)
                    atoms.append(Atom(instruction["op"], instruction["attribute"], value))
                program = Program(tuple(atoms))
                if invalid_program(program):
                    return ()
                parsed.append(program)
            return tuple(parsed)
        except (ValueError, KeyError, TypeError):
            return ()
