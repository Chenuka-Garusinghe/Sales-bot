"""
Router for Stage 1: Lead Agent endpoints.

All endpoints are prefixed with /api/leads (set in main.py).

Endpoints:
    GET  /api/leads/tools         -- list available lead source tools
    POST /api/leads/run           -- run the LangGraph agent with a prompt
    GET  /api/leads/leads/{id}    -- retrieve a cached pipeline run
"""

from fastapi import APIRouter, HTTPException

from app.models.leads import LeadAgentPipelineResponse, LeadRunRequest
from app.services.lead_agent import TOOLS, get_cached_run, run_pipeline

router = APIRouter()


@router.get("/tools")
async def list_tools():
    """List all lead source tools the agent can call.
    Returns each tool's name and description (what the LLM sees).
    """
    return [
        {"name": t.name, "description": t.description}
        for t in TOOLS
    ]


@router.post("/run", response_model=LeadAgentPipelineResponse)
async def execute_pipeline(request: LeadRunRequest | None = None):
    """Run the lead gathering pipeline.

    Send a natural language prompt and the LangGraph agent will:
    1. Decide which lead sources to search
    2. Call the appropriate tools with filters
    3. Pass results through deterministic evaluation
    4. Return qualified leads with the agent's summary

    Example request body:
        {"prompt": "Find fintech companies with 100+ employees", "min_company_size": 100}
    """
    req = request or LeadRunRequest()
    return await run_pipeline(
        prompt=req.prompt,
        min_company_size=req.min_company_size,
    )


@router.get("/leads/{run_id}", response_model=LeadAgentPipelineResponse)
async def get_run_results(run_id: str):
    """Retrieve results from a previous pipeline run using its run_id.
    The run_id is returned in the response of POST /run.
    Results are stored in memory and lost on server restart.
    """
    result = get_cached_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return result
