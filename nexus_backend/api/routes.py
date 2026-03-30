# api/routes.py
from fastapi import APIRouter, HTTPException
from .schemas import DetectionResponse, InvestigationRequest, SARResponse
from services.gnn_inference import detector
from services.agent_workflow import generate_sar_report

router = APIRouter()

@router.get("/detect", response_model=DetectionResponse)
async def run_detection():
    """Triggers the GNN to scan the entire graph."""
    try:
        count, scores = detector.detect_anomalies(threshold=0.90)
        return DetectionResponse(
            flagged_nodes=list(scores.keys()),
            risk_scores=scores,
            message=f"Scan complete. {count} high-risk nodes detected."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/investigate", response_model=SARResponse)
async def run_investigation(req: InvestigationRequest):
    """Triggers the Agentic AI to write a report on a specific node."""
    try:
        # Await the LLM generation so the event loop isn't blocked
        report = await generate_sar_report(req.node_id, req.hop_depth)
        
        return SARResponse(
            node_id=req.node_id,
            sar_report=report,
            agents_involved=["Data Agent", "Context Agent", "Reasoning Agent"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))