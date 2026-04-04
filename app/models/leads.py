from pydantic import BaseModel
from enum import Enum
from datetime import datetime


class LeadSource(str, Enum):
    APOLLO = "apollo"
    LINKEDIN = "linkedin"
    WEB_SCRAPE = "web_scrape"
    REFERRAL = "referral"


class Lead(BaseModel):
    id: str
    name: str
    company: str
    title: str
    email: str
    source: LeadSource
    industry: str
    company_size: int
    relevance_score: float = 0.0
    is_authentic: bool = True
    rejection_reason: str | None = None


class AgentResult(BaseModel):
    agent_id: str
    source: LeadSource
    leads_found: list[Lead]
    search_duration_ms: int
    timestamp: datetime


class EvaluationResult(BaseModel):
    total_gathered: int
    passed_filter: int
    rejected: int
    rejection_breakdown: dict[str, int]
    qualified_leads: list[Lead]


class OpenClawPipelineResponse(BaseModel):
    run_id: str
    agent_results: list[AgentResult]
    evaluation: EvaluationResult
    total_duration_ms: int


class OpenClawRunRequest(BaseModel):
    industry_filter: str | None = None
    min_company_size: int = 50
