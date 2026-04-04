from fastapi import APIRouter, HTTPException

from app.models.leads import LeadAgentPipelineResponse, LeadRunRequest
from app.services.lead_agent import TOOLS, get_cached_run, run_pipeline

router = APIRouter()


@router.get("/tools")
async def list_tools():
    return [
        {"name": t.name, "description": t.description}
        for t in TOOLS
    ]


@router.post("/run", response_model=LeadAgentPipelineResponse)
async def execute_pipeline(request: LeadRunRequest | None = None):
    req = request or LeadRunRequest()
    return await run_pipeline(
        prompt=req.prompt,
        min_company_size=req.min_company_size,
    )


@router.get("/leads/{run_id}", response_model=LeadAgentPipelineResponse)
async def get_run_results(run_id: str):
    result = get_cached_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return result
