"""
Data models for Stage 1: Lead Agent pipeline.

These Pydantic models define the shape of all request/response data
for the LangGraph-based lead gathering and evaluation system.
"""

from pydantic import BaseModel
from enum import Enum


class LeadSource(str, Enum):
    """The four lead sources the agent can search.
    Each maps to a tool in app/services/lead_agent.py and
    a mock dataset in app/mock_data/leads.py.
    """
    APOLLO = "apollo"
    LINKEDIN = "linkedin"
    WEB_SCRAPE = "web_scrape"
    REFERRAL = "referral"


class Lead(BaseModel):
    """A single sales lead gathered from any source.

    Fields set during gathering:
        id, name, company, title, email, source, industry, company_size, is_authentic

    Fields set during evaluation (by lead_evaluator.py):
        relevance_score  -- 0.0 to 1.0, based on title seniority + company size + industry
        rejection_reason -- set if the lead was filtered out, None if it passed
    """
    id: str
    name: str
    company: str
    title: str
    email: str
    source: LeadSource
    industry: str
    company_size: int
    relevance_score: float = 0.0        # set by evaluator, not by tools
    is_authentic: bool = True            # some mock leads are intentionally fake
    rejection_reason: str | None = None  # e.g. "duplicate", "invalid_email", etc.


class EvaluationResult(BaseModel):
    """Output of the deterministic lead evaluator (lead_evaluator.py).
    Contains both summary stats and the filtered/scored list of qualified leads.
    """
    total_gathered: int                     # how many raw leads came in from all tools
    passed_filter: int                      # how many survived all filter rules
    rejected: int                           # how many were rejected
    rejection_breakdown: dict[str, int]     # reason -> count (e.g. {"duplicate": 2, "invalid_email": 1})
    qualified_leads: list[Lead]             # sorted by relevance_score descending


class ToolCallRecord(BaseModel):
    """Logs which tool the LangGraph agent called and with what arguments.
    Useful for understanding the agent's reasoning in the response.
    """
    tool_name: str      # e.g. "search_apollo"
    arguments: dict     # e.g. {"industry": "fintech", "min_company_size": 100}


class LeadAgentPipelineResponse(BaseModel):
    """The full response from POST /api/leads/run.

    Contains everything: what the agent did (tool_calls, agent_summary),
    and the final evaluated results (evaluation).
    """
    run_id: str                         # short UUID for caching/retrieval
    prompt: str                         # the original natural language query
    tool_calls: list[ToolCallRecord]    # which tools the agent chose to call
    agent_summary: str                  # the agent's natural language summary of findings
    evaluation: EvaluationResult        # deterministic filter/score results
    total_duration_ms: int              # end-to-end pipeline time


class LeadRunRequest(BaseModel):
    """Request body for POST /api/leads/run.

    prompt: Natural language query like "find fintech leads in Sydney"
            The LangGraph agent interprets this and decides which tools to call.
    min_company_size: Passed to the deterministic evaluator (not the agent).
    """
    prompt: str = "Find leads across all available sources"
    min_company_size: int = 50
