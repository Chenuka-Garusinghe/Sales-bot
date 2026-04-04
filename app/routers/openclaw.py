from fastapi import APIRouter, HTTPException

from app.models.leads import OpenClawPipelineResponse, OpenClawRunRequest
from app.services.openclaw import AGENT_SOURCE_MAP, get_cached_run, run_pipeline

router = APIRouter()


@router.get("/agents")
async def list_agents():
    return [
        {"agent_id": agent_id, "source": source.value}
        for agent_id, (source, _) in AGENT_SOURCE_MAP.items()
    ]


@router.post("/run", response_model=OpenClawPipelineResponse)
async def execute_pipeline(request: OpenClawRunRequest | None = None):
    req = request or OpenClawRunRequest()
    return await run_pipeline(
        industry_filter=req.industry_filter,
        min_company_size=req.min_company_size,
    )


@router.get("/leads/{run_id}", response_model=OpenClawPipelineResponse)
async def get_run_results(run_id: str):
    result = get_cached_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return result
