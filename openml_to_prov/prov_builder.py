"""W3C PROV-JSON document builder."""

import json
from typing import Any, Dict, Optional


class ProvDocumentBuilder:
    """Builds W3C PROV-JSON documents."""

    # Short blank-node prefix per relation type (kept stable for diffability)
    _BNODE_PREFIX = {
        "used": "u",
        "wasGeneratedBy": "g",
        "wasAssociatedWith": "a",
        "wasInformedBy": "i",
        "wasAttributedTo": "t",
        "wasDerivedFrom": "d",
    }

    def __init__(self):
        self.doc = {
            "prefix": {
                "prov": "http://www.w3.org/ns/prov#",
                "openml": "https://openml.org/def/",
                "ml": "https://ml-schema.org/",
                "e": "urn:entity:",
                "a": "urn:activity:",
                "ag": "urn:agent:",
                "checksum": "urn:checksum:",
                "fold": "urn:fold:",
            },
            "entity": {},
            "activity": {},
            "agent": {},
            "used": {},
            "wasGeneratedBy": {},
            "wasAssociatedWith": {},
            "wasInformedBy": {},
            "wasAttributedTo": {},
            "wasDerivedFrom": {},
        }
        self._rel_counters = {k: 0 for k in self._BNODE_PREFIX}

    def _next_rid(self, rel: str) -> str:
        """Generate a unique blank-node identifier for a relation."""
        self._rel_counters[rel] += 1
        return f"_:{self._BNODE_PREFIX[rel]}{self._rel_counters[rel]}"

    # ---- Elements -----------------------------------------------------------

    def add_entity(self, entity_id: str, attributes: Dict[str, Any]):
        """Add an entity to the document."""
        self.doc["entity"][entity_id] = attributes

    def add_activity(self, activity_id: str, attributes: Dict[str, Any]):
        """Add an activity to the document."""
        self.doc["activity"][activity_id] = attributes

    def add_agent(self, agent_id: str, attributes: Dict[str, Any]):
        """Add an agent to the document."""
        self.doc["agent"][agent_id] = attributes

    # ---- Relations ----------------------------------------------------------

    def add_used(self, activity: str, entity: str, role: Optional[str] = None):
        """Add a 'used' relation."""
        rel = {"prov:activity": activity, "prov:entity": entity}
        if role:
            rel["prov:role"] = role
        self.doc["used"][self._next_rid("used")] = rel

    def add_generation(self, entity: str, activity: str):
        """Add a 'wasGeneratedBy' relation."""
        self.doc["wasGeneratedBy"][self._next_rid("wasGeneratedBy")] = {
            "prov:entity": entity,
            "prov:activity": activity,
        }

    def add_association(self, activity: str, agent: str):
        """Add a 'wasAssociatedWith' relation."""
        self.doc["wasAssociatedWith"][self._next_rid("wasAssociatedWith")] = {
            "prov:activity": activity,
            "prov:agent": agent,
        }

    def add_communication(self, informed: str, informant: str):
        """Add a 'wasInformedBy' relation."""
        self.doc["wasInformedBy"][self._next_rid("wasInformedBy")] = {
            "prov:informed": informed,
            "prov:informant": informant,
        }

    def add_attribution(self, entity: str, agent: str):
        """Add a 'wasAttributedTo' relation."""
        self.doc["wasAttributedTo"][self._next_rid("wasAttributedTo")] = {
            "prov:entity": entity,
            "prov:agent": agent,
        }

    def add_derivation(self, generated: str, used: str):
        """Add a 'wasDerivedFrom' relation."""
        self.doc["wasDerivedFrom"][self._next_rid("wasDerivedFrom")] = {
            "prov:generatedEntity": generated,
            "prov:usedEntity": used,
        }

    # ---- Reporting / serialisation ------------------------------------------

    def graph_stats(self) -> Dict[str, int]:
        """Return node and edge counts for this document."""
        n_entities = len(self.doc["entity"])
        n_activities = len(self.doc["activity"])
        n_agents = len(self.doc["agent"])
        edge_relations = list(self._BNODE_PREFIX)
        stats = {
            "nodes": n_entities + n_activities + n_agents,
            "entities": n_entities,
            "activities": n_activities,
            "agents": n_agents,
            "edges": sum(len(self.doc[r]) for r in edge_relations),
        }
        stats.update({r: len(self.doc[r]) for r in edge_relations})
        return stats

    def to_json(self, pretty: bool = True) -> str:
        """Serialize document to JSON string."""
        if pretty:
            return json.dumps(self.doc, indent=2, default=str)
        return json.dumps(self.doc, separators=(",", ":"), default=str)
