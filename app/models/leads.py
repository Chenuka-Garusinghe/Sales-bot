from pydantic import BaseModel
from enum import Enum


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


class EvaluationResult(BaseModel):
    total_gathered: int
    passed_filter: int
    rejected: int
    rejection_breakdown: dict[str, int]
    qualified_leads: list[Lead]


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict


class LeadAgentPipelineResponse(BaseModel):
    run_id: str
    prompt: str
    tool_calls: list[ToolCallRecord]
    agent_summary: str
    evaluation: EvaluationResult
    total_duration_ms: int


class LeadRunRequest(BaseModel):
    prompt: str = "Find leads across all available sources"
    min_company_size: int = 50
