import json
import os
import time
import uuid
from typing import Annotated

import operator
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.mock_data.leads import (
    APOLLO_LEADS,
    LINKEDIN_LEADS,
    REFERRAL_LEADS,
    WEB_SCRAPE_LEADS,
)
from app.models.leads import (
    EvaluationResult,
    Lead,
    LeadAgentPipelineResponse,
    ToolCallRecord,
)
from app.services.lead_evaluator import evaluate_leads

# ── Tools (one per lead source) ──────────────────────────────────────────


@tool
def search_apollo(industry: str = "", min_company_size: int = 0) -> str:
    """Search the Apollo lead database for B2B contacts.

    Args:
        industry: Optional industry filter (e.g. 'fintech', 'healthcare', 'saas', 'education', 'professional_services').
        min_company_size: Optional minimum company size to filter by.
    """
    leads = [Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None}) for l in APOLLO_LEADS]
    if industry:
        leads = [l for l in leads if l.industry.lower() == industry.lower()]
    if min_company_size:
        leads = [l for l in leads if l.company_size >= min_company_size]
    return json.dumps([l.model_dump() for l in leads], default=str)


@tool
def search_linkedin(industry: str = "", min_company_size: int = 0) -> str:
    """Search LinkedIn for professional contacts and potential leads.

    Args:
        industry: Optional industry filter (e.g. 'fintech', 'healthcare', 'saas', 'education', 'professional_services').
        min_company_size: Optional minimum company size to filter by.
    """
    leads = [Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None}) for l in LINKEDIN_LEADS]
    if industry:
        leads = [l for l in leads if l.industry.lower() == industry.lower()]
    if min_company_size:
        leads = [l for l in leads if l.company_size >= min_company_size]
    return json.dumps([l.model_dump() for l in leads], default=str)


@tool
def search_web(industry: str = "", min_company_size: int = 0) -> str:
    """Scrape the public web for potential business leads from company websites and directories.

    Args:
        industry: Optional industry filter (e.g. 'fintech', 'healthcare', 'saas', 'education', 'professional_services').
        min_company_size: Optional minimum company size to filter by.
    """
    leads = [Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None}) for l in WEB_SCRAPE_LEADS]
    if industry:
        leads = [l for l in leads if l.industry.lower() == industry.lower()]
    if min_company_size:
        leads = [l for l in leads if l.company_size >= min_company_size]
    return json.dumps([l.model_dump() for l in leads], default=str)


@tool
def search_referrals(industry: str = "", min_company_size: int = 0) -> str:
    """Search the partner referral network for warm leads from existing relationships.

    Args:
        industry: Optional industry filter (e.g. 'fintech', 'healthcare', 'saas', 'education', 'professional_services').
        min_company_size: Optional minimum company size to filter by.
    """
    leads = [Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None}) for l in REFERRAL_LEADS]
    if industry:
        leads = [l for l in leads if l.industry.lower() == industry.lower()]
    if min_company_size:
        leads = [l for l in leads if l.company_size >= min_company_size]
    return json.dumps([l.model_dump() for l in leads], default=str)


TOOLS = [search_apollo, search_linkedin, search_web, search_referrals]

# ── LangGraph state & graph ──────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def _build_graph():
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    model = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Pipeline runner ──────────────────────────────────────────────────────

_run_cache: dict[str, LeadAgentPipelineResponse] = {}

SYSTEM_PROMPT = (
    "You are a sales lead research agent for SimplAI AU. "
    "You have access to four lead source tools: search_apollo, search_linkedin, "
    "search_web, and search_referrals. Given the user's request, decide which "
    "sources are most relevant and call them with appropriate filters. "
    "You can call multiple tools if needed. Pass industry and min_company_size "
    "filters when the user's query implies them. "
    "After receiving tool results, provide a brief summary of what you found."
)


async def run_pipeline(prompt: str, min_company_size: int = 50) -> LeadAgentPipelineResponse:
    start = time.monotonic()
    run_id = str(uuid.uuid4())[:8]

    graph = _build_graph()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        HumanMessage(content=prompt),
    ]

    result = await graph.ainvoke({"messages": messages})

    # Extract leads from tool call results
    all_leads: list[Lead] = []
    tool_calls_log: list[ToolCallRecord] = []

    for msg in result["messages"]:
        # Collect tool call records from the AI message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append(
                    ToolCallRecord(
                        tool_name=tc["name"],
                        arguments=tc["args"],
                    )
                )
        # Parse leads from tool result messages
        if msg.type == "tool" and msg.content:
            try:
                raw_leads = json.loads(msg.content)
                for ld in raw_leads:
                    all_leads.append(Lead(**ld))
            except (json.JSONDecodeError, TypeError):
                pass

    # Get agent's final summary
    agent_summary = ""
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and isinstance(msg.content, str) and msg.content:
            agent_summary = msg.content
            break

    evaluation: EvaluationResult = evaluate_leads(all_leads, min_company_size)
    total_ms = int((time.monotonic() - start) * 1000)

    response = LeadAgentPipelineResponse(
        run_id=run_id,
        prompt=prompt,
        tool_calls=tool_calls_log,
        agent_summary=agent_summary,
        evaluation=evaluation,
        total_duration_ms=total_ms,
    )

    _run_cache[run_id] = response
    return response


def get_cached_run(run_id: str) -> LeadAgentPipelineResponse | None:
    return _run_cache.get(run_id)
