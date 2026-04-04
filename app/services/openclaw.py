import asyncio
import random
import time
import uuid
from datetime import datetime, timezone

from app.mock_data.leads import (
    APOLLO_LEADS,
    LINKEDIN_LEADS,
    REFERRAL_LEADS,
    WEB_SCRAPE_LEADS,
)
from app.models.leads import (
    AgentResult,
    EvaluationResult,
    Lead,
    LeadSource,
    OpenClawPipelineResponse,
)
from app.services.lead_evaluator import evaluate_leads

AGENT_SOURCE_MAP: dict[str, tuple[LeadSource, list[dict]]] = {
    "agent-apollo": (LeadSource.APOLLO, APOLLO_LEADS),
    "agent-linkedin": (LeadSource.LINKEDIN, LINKEDIN_LEADS),
    "agent-webscrape": (LeadSource.WEB_SCRAPE, WEB_SCRAPE_LEADS),
    "agent-referral": (LeadSource.REFERRAL, REFERRAL_LEADS),
}

# In-memory cache for pipeline runs
_run_cache: dict[str, OpenClawPipelineResponse] = {}


async def _run_agent(
    agent_id: str,
    source: LeadSource,
    raw_leads: list[dict],
    industry_filter: str | None = None,
) -> AgentResult:
    delay_ms = random.randint(200, 800)
    await asyncio.sleep(delay_ms / 1000)

    leads = [Lead(**{**lead, "relevance_score": 0.0, "rejection_reason": None}) for lead in raw_leads]

    if industry_filter:
        leads = [l for l in leads if l.industry.lower() == industry_filter.lower()]

    return AgentResult(
        agent_id=agent_id,
        source=source,
        leads_found=leads,
        search_duration_ms=delay_ms,
        timestamp=datetime.now(timezone.utc),
    )


async def run_pipeline(
    industry_filter: str | None = None,
    min_company_size: int = 50,
) -> OpenClawPipelineResponse:
    start = time.monotonic()
    run_id = str(uuid.uuid4())[:8]

    tasks = [
        _run_agent(agent_id, source, leads, industry_filter)
        for agent_id, (source, leads) in AGENT_SOURCE_MAP.items()
    ]
    agent_results: list[AgentResult] = await asyncio.gather(*tasks)

    all_leads: list[Lead] = []
    for result in agent_results:
        all_leads.extend(result.leads_found)

    evaluation: EvaluationResult = evaluate_leads(all_leads, min_company_size)
    total_ms = int((time.monotonic() - start) * 1000)

    response = OpenClawPipelineResponse(
        run_id=run_id,
        agent_results=agent_results,
        evaluation=evaluation,
        total_duration_ms=total_ms,
    )

    _run_cache[run_id] = response
    return response


def get_cached_run(run_id: str) -> OpenClawPipelineResponse | None:
    return _run_cache.get(run_id)
