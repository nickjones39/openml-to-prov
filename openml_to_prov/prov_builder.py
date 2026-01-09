"""W3C PROV-JSON document builder."""

import json
from typing import Dict, Any, Optional


class ProvDocumentBuilder:
    """Builds W3C PROV-JSON documents."""

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
            "used": [],
            "wasGeneratedBy": [],
            "wasAssociatedWith": [],
            "wasInformedBy": [],
            "wasAttributedTo": [],
            "wasDerivedFrom": [],
        }

    def add_entity(self, entity_id: str, attributes: Dict[str, Any]):
        """Add an entity to the document."""
        self.doc["entity"][entity_id] = attributes

    def add_activity(self, activity_id: str, attributes: Dict[str, Any]):
        """Add an activity to the document."""
        self.doc["activity"][activity_id] = attributes

    def add_agent(self, agent_id: str, attributes: Dict[str, Any]):
        """Add an agent to the document."""
        self.doc["agent"][agent_id] = attributes

    def add_used(self, activity: str, entity: str, role: Optional[str] = None):
        """Add a 'used' relation."""
        rel = {"activity": activity, "entity": entity}
        if role:
            rel["prov:role"] = role
        self.doc["used"].append(rel)

    def add_generation(self, entity: str, activity: str):
        """Add a 'wasGeneratedBy' relation."""
        self.doc["wasGeneratedBy"].append({"entity": entity, "activity": activity})

    def add_association(self, activity: str, agent: str):
        """Add a 'wasAssociatedWith' relation."""
        self.doc["wasAssociatedWith"].append({"activity": activity, "agent": agent})

    def add_communication(self, informed: str, informant: str):
        """Add a 'wasInformedBy' relation."""
        self.doc["wasInformedBy"].append({"informed": informed, "informant": informant})

    def add_attribution(self, entity: str, agent: str):
        """Add a 'wasAttributedTo' relation."""
        self.doc["wasAttributedTo"].append({"entity": entity, "agent": agent})

    def add_derivation(self, generated: str, used: str):
        """Add a 'wasDerivedFrom' relation."""
        self.doc["wasDerivedFrom"].append({"generatedEntity": generated, "usedEntity": used})

    def to_json(self, pretty: bool = True) -> str:
        """Serialize document to JSON string."""
        if pretty:
            return json.dumps(self.doc, indent=2, default=str)
        return json.dumps(self.doc, separators=(',', ':'), default=str)
