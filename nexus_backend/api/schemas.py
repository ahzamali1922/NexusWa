# api/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

class DetectionResponse(BaseModel):
    flagged_nodes: List[int]
    risk_scores: Dict[int, float]
    message: str

class InvestigationRequest(BaseModel):
    node_id: int
    hop_depth: int = 2  # How far out in the graph to look

class SARResponse(BaseModel):
    node_id: int
    sar_report: str
    agents_involved: List[str]