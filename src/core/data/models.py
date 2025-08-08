from typing import Dict, List, Optional

from pydantic import BaseModel


class LinguisticGeoid(BaseModel):
    primary_statement: str
    confidence_score: float
    source_geoid_id: str
    supporting_scars: List[Dict] = []
    potential_ambiguities: List[str] = []
    explanation_lineage: str
