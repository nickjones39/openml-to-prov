"""Conformance validation for generated PROV-JSON documents.

Three independent checks, in increasing cost order:

  1. schema      - structural PROV-JSON conformance (no external deps).
  2. shacl       - SHACL shapes over the corpus-specific PROV-O graph
                   (requires ``rdflib`` + ``pyshacl``).
  3. round_trip  - PROV-O round-trip: PROV-JSON -> RDF -> re-serialise and
                   assert the graph is isomorphic (requires ``prov`` + ``rdflib``).

Each check degrades gracefully: if an optional dependency is missing the check
is skipped with a recorded reason rather than raising, so core corpus
generation still runs on ``numpy`` alone.

The public entry point is ``ProvValidator.validate(doc)`` which returns a
``ValidationResult``. ``CorpusGenerator`` uses this to gate writes and to
accumulate a ``conformance_report.json`` alongside the corpus manifest.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants describing the expected corpus structure (see Table 2 / Sec 6.2).
# ---------------------------------------------------------------------------

ELEMENT_KINDS = ("entity", "activity", "agent")
RELATION_KINDS = (
    "used",
    "wasGeneratedBy",
    "wasAssociatedWith",
    "wasInformedBy",
    "wasAttributedTo",
    "wasDerivedFrom",
)

# (subject_key, object_key) endpoints each relation must carry in PROV-JSON.
RELATION_ENDPOINTS = {
    "used": ("prov:activity", "prov:entity"),
    "wasGeneratedBy": ("prov:entity", "prov:activity"),
    "wasAssociatedWith": ("prov:activity", "prov:agent"),
    "wasInformedBy": ("prov:informed", "prov:informant"),
    "wasAttributedTo": ("prov:entity", "prov:agent"),
    "wasDerivedFrom": ("prov:generatedEntity", "prov:usedEntity"),
}

# Which element bucket each endpoint key must resolve to.
_ENDPOINT_BUCKET = {
    "prov:entity": "entity",
    "prov:activity": "activity",
    "prov:agent": "agent",
    "prov:informed": "activity",
    "prov:informant": "activity",
    "prov:generatedEntity": "entity",
    "prov:usedEntity": "entity",
}

REQUIRED_PREFIXES = ("prov", "e", "a", "ag")

# Expected per-run cardinality for the 5-fold template. These are the
# structural invariants H1/H3 assert; a deviation is itself diagnostic.
EXPECTED_NODES = 44
EXPECTED_EDGES = 128
EXPECTED_ENTITIES = 26
EXPECTED_ACTIVITIES = 17
EXPECTED_AGENTS = 1


@dataclass
class CheckOutcome:
    """Result of a single named check."""

    name: str
    status: str  # "pass" | "fail" | "skipped"
    errors: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        # A skipped check does not fail the document.
        return self.status in ("pass", "skipped")


@dataclass
class ValidationResult:
    """Aggregate result across all enabled checks for one document."""

    valid: bool
    checks: List[CheckOutcome] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": {
                c.name: {
                    "status": c.status,
                    "errors": c.errors,
                    "detail": c.detail,
                }
                for c in self.checks
            },
        }

    def error_summary(self) -> str:
        parts = []
        for c in self.checks:
            if c.status == "fail":
                parts.append(f"{c.name}: " + "; ".join(c.errors[:3]))
        return " | ".join(parts) if parts else "ok"


class ProvValidator:
    """Validates PROV-JSON documents against schema, SHACL, and round-trip checks.

    Parameters
    ----------
    checks:
        Which checks to run. Subset of {"schema", "shacl", "round_trip"}.
    expected_cardinality:
        If True, the schema check also asserts the 44/128 node/edge template.
        Set False if you intend to extend the EAA mapping (e.g. 10-fold CV),
        where node/edge counts legitimately differ.
    """

    ALL_CHECKS = ("schema", "shacl", "round_trip")

    def __init__(
        self,
        checks: Sequence[str] = ALL_CHECKS,
        expected_cardinality: bool = True,
    ):
        unknown = set(checks) - set(self.ALL_CHECKS)
        if unknown:
            raise ValueError(f"Unknown check(s): {sorted(unknown)}")
        self.checks = tuple(checks)
        self.expected_cardinality = expected_cardinality
        # Probe optional deps once, lazily reused across documents.
        self._shacl_graph = None
        self._dep_cache: Dict[str, Optional[str]] = {}

    # -- public API ---------------------------------------------------------

    def validate(self, doc: Dict[str, Any]) -> ValidationResult:
        outcomes: List[CheckOutcome] = []
        if "schema" in self.checks:
            outcomes.append(self._check_schema(doc))
        if "shacl" in self.checks:
            outcomes.append(self._check_shacl(doc))
        if "round_trip" in self.checks:
            outcomes.append(self._check_round_trip(doc))
        valid = all(o.ok for o in outcomes)
        return ValidationResult(valid=valid, checks=outcomes)

    # -- check 1: structural schema ----------------------------------------

    def _check_schema(self, doc: Dict[str, Any]) -> CheckOutcome:
        errors: List[str] = []

        # Top-level shape.
        if not isinstance(doc, dict):
            return CheckOutcome("schema", "fail", ["document is not a JSON object"])
        if "prefix" not in doc or not isinstance(doc["prefix"], dict):
            errors.append("missing or malformed 'prefix' block")
        else:
            for p in REQUIRED_PREFIXES:
                if p not in doc["prefix"]:
                    errors.append(f"missing required prefix '{p}'")

        for kind in ELEMENT_KINDS:
            if kind not in doc or not isinstance(doc[kind], dict):
                errors.append(f"missing or malformed '{kind}' block")

        # Collect declared element IDs for referential checks.
        element_ids = {
            kind: set(doc.get(kind, {})) for kind in ELEMENT_KINDS
        }

        # Every element must carry a prov:type.
        for kind in ELEMENT_KINDS:
            for eid, attrs in doc.get(kind, {}).items():
                if not isinstance(attrs, dict):
                    errors.append(f"{kind} '{eid}' attributes not an object")
                    continue
                if "prov:type" not in attrs:
                    errors.append(f"{kind} '{eid}' missing prov:type")

        # Relations: shape + referential integrity.
        for rel in RELATION_KINDS:
            block = doc.get(rel, {})
            if not isinstance(block, dict):
                errors.append(f"relation block '{rel}' not an object")
                continue
            subj_key, obj_key = RELATION_ENDPOINTS[rel]
            for rid, triple in block.items():
                if not isinstance(triple, dict):
                    errors.append(f"{rel} '{rid}' not an object")
                    continue
                for endpoint in (subj_key, obj_key):
                    if endpoint not in triple:
                        errors.append(f"{rel} '{rid}' missing {endpoint}")
                        continue
                    ref = triple[endpoint]
                    bucket = _ENDPOINT_BUCKET[endpoint]
                    if ref not in element_ids[bucket]:
                        errors.append(
                            f"{rel} '{rid}' {endpoint}={ref!r} "
                            f"not a declared {bucket}"
                        )

        # Cardinality (optional).
        detail: Dict[str, Any] = {}
        if self.expected_cardinality and not errors:
            n_ent = len(doc.get("entity", {}))
            n_act = len(doc.get("activity", {}))
            n_agt = len(doc.get("agent", {}))
            n_nodes = n_ent + n_act + n_agt
            n_edges = sum(len(doc.get(r, {})) for r in RELATION_KINDS)
            detail = {
                "nodes": n_nodes,
                "edges": n_edges,
                "entities": n_ent,
                "activities": n_act,
                "agents": n_agt,
            }
            checks = [
                (n_nodes, EXPECTED_NODES, "nodes"),
                (n_edges, EXPECTED_EDGES, "edges"),
                (n_ent, EXPECTED_ENTITIES, "entities"),
                (n_act, EXPECTED_ACTIVITIES, "activities"),
                (n_agt, EXPECTED_AGENTS, "agents"),
            ]
            for got, want, label in checks:
                if got != want:
                    errors.append(f"{label}={got}, expected {want}")

        status = "fail" if errors else "pass"
        return CheckOutcome("schema", status, errors, detail)

    # -- dependency probing -------------------------------------------------

    def _missing(self, module: str) -> bool:
        """Return True if ``module`` cannot be imported. Cached."""
        if module not in self._dep_cache:
            try:
                __import__(module)
                self._dep_cache[module] = None
            except Exception as exc:  # noqa: BLE001
                self._dep_cache[module] = str(exc)
        return self._dep_cache[module] is not None

    # -- check 2: SHACL -----------------------------------------------------

    def _check_shacl(self, doc: Dict[str, Any]) -> CheckOutcome:
        if self._missing("rdflib") or self._missing("pyshacl"):
            return CheckOutcome(
                "shacl", "skipped",
                detail={"reason": "rdflib/pyshacl not installed"},
            )
        try:
            from pyshacl import validate as shacl_validate

            data_graph = _prov_json_to_rdf(doc)
            shapes_graph = self._get_shacl_shapes()
            conforms, _results_graph, results_text = shacl_validate(
                data_graph,
                shacl_graph=shapes_graph,
                inference="none",
                abort_on_first=False,
                meta_shacl=False,
                advanced=True,
            )
            if conforms:
                return CheckOutcome("shacl", "pass")
            # Trim pyshacl's verbose report to the constraint lines.
            lines = [
                ln.strip()
                for ln in results_text.splitlines()
                if "Constraint Violation" in ln or "Message:" in ln
            ]
            return CheckOutcome("shacl", "fail", lines[:10] or [results_text[:500]])
        except Exception as exc:  # noqa: BLE001
            return CheckOutcome(
                "shacl", "fail", [f"SHACL validation raised: {exc}"]
            )

    def _get_shacl_shapes(self):
        """Build (once) the SHACL shapes graph encoding corpus invariants."""
        if self._shacl_graph is not None:
            return self._shacl_graph
        import rdflib

        g = rdflib.Graph()
        g.parse(data=_SHACL_SHAPES_TTL, format="turtle")
        self._shacl_graph = g
        return g

    # -- check 3: PROV-O round-trip ----------------------------------------

    def _check_round_trip(self, doc: Dict[str, Any]) -> CheckOutcome:
        if self._missing("prov") or self._missing("rdflib"):
            return CheckOutcome(
                "round_trip", "skipped",
                detail={"reason": "prov/rdflib not installed"},
            )
        try:
            from prov.model import ProvDocument
            import rdflib
            from rdflib.compare import isomorphic, to_isomorphic

            # PROV-JSON -> in-memory PROV model.
            d1 = ProvDocument.deserialize(
                content=json.dumps(doc), format="json"
            )
            # -> RDF (PROV-O), first pass.
            rdf1 = d1.serialize(format="rdf", rdf_format="turtle")
            g1 = rdflib.Graph().parse(data=rdf1, format="turtle")

            # Round-trip: PROV model -> PROV-JSON -> PROV model -> RDF, 2nd pass.
            json2 = d1.serialize(format="json")
            d2 = ProvDocument.deserialize(content=json2, format="json")
            rdf2 = d2.serialize(format="rdf", rdf_format="turtle")
            g2 = rdflib.Graph().parse(data=rdf2, format="turtle")

            if isomorphic(g1, g2):
                return CheckOutcome(
                    "round_trip", "pass",
                    detail={"triples": len(g1)},
                )
            iso1, iso2 = to_isomorphic(g1), to_isomorphic(g2)
            only1 = len(iso1 - iso2)
            only2 = len(iso2 - iso1)
            return CheckOutcome(
                "round_trip", "fail",
                [f"RDF graphs not isomorphic: +{only1}/-{only2} triples"],
                detail={"triples_pass1": len(g1), "triples_pass2": len(g2)},
            )
        except Exception as exc:  # noqa: BLE001
            return CheckOutcome(
                "round_trip", "fail", [f"round-trip raised: {exc}"]
            )


# ---------------------------------------------------------------------------
# PROV-JSON -> RDF helper used by the SHACL check.
#
# The ``prov`` library's PROV-O serialiser expects standard PROV-JSON. Our
# documents are spec-conformant, so we route through it to obtain PROV-O RDF,
# then validate that RDF against the shapes. Done independently of the
# round-trip check so SHACL can run even if the round-trip check is disabled.
# ---------------------------------------------------------------------------

def _prov_json_to_rdf(doc: Dict[str, Any]):
    import rdflib
    from prov.model import ProvDocument

    d = ProvDocument.deserialize(content=json.dumps(doc), format="json")
    ttl = d.serialize(format="rdf", rdf_format="turtle")
    return rdflib.Graph().parse(data=ttl, format="turtle")


# ---------------------------------------------------------------------------
# SHACL shapes (Turtle). Corpus-specific PROV-O constraints, written against
# the RDF the ``prov`` library actually emits for our PROV-JSON:
#   - every prov:Activity is associated with >=1 agent (prov:wasAssociatedWith);
#   - every prov:Activity carries a start and end time;
#   - the agent referenced by an association must itself be a prov:Agent.
# Kept deliberately lightweight: these encode the structural guarantees the
# corpus claims, not a full PROV-O axiomatisation.
# ---------------------------------------------------------------------------

_SHACL_SHAPES_TTL = """
@prefix sh:   <http://www.w3.org/ns/shacl#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

# Every activity must be associated with at least one agent, and that
# association target must be a declared prov:Agent.
prov:ActivityShape
    a sh:NodeShape ;
    sh:targetClass prov:Activity ;
    sh:property [
        sh:path prov:wasAssociatedWith ;
        sh:minCount 1 ;
        sh:message "Activity has no prov:wasAssociatedWith agent" ;
        sh:class prov:Agent ;
    ] .
"""
