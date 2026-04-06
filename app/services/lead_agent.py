"""
Stage 1: LangGraph ReAct Lead Agent

This is the core of the lead gathering pipeline. It uses a LangGraph StateGraph
to build a ReAct (Reasoning + Acting) agent that:

1. Receives a natural language prompt from a sales manager
2. Reasons about which lead sources to search (using an LLM -- Ollama/llama3.1)
3. Calls the appropriate tools with filters the LLM decides on
4. Passes all gathered leads to the deterministic evaluator
5. Returns the full pipeline response

GRAPH FLOW:
    START → agent (LLM decides what tools to call)
          → tools (executes the tool calls, returns results)
          → agent (LLM summarises findings)
          → END

The agent node and tools node loop until the LLM stops making tool calls.

TO SWAP THE LLM:
    Change the ChatOllama() call in _build_graph() to any LangChain chat model.
    e.g. ChatAnthropic(model="claude-sonnet-4-20250514", api_key=...)
         ChatOpenAI(model="gpt-4o", api_key=...)
"""

import json
import operator
import time
import uuid
from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
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


# ── Tools ────────────────────────────────────────────────────────────────
# Each tool wraps a mock lead source. The @tool decorator registers it
# with LangGraph so the LLM can call it. The docstring is what the LLM
# reads to understand what the tool does.
#
# In production, these would make real API calls to Apollo, LinkedIn, etc.
# Right now they just filter the hardcoded mock data from app/mock_data/leads.py.


@tool
def search_apollo(industry: str = "", min_company_size: int = 0) -> str:
    """Search the Apollo lead database for B2B contacts.

    Args:
        industry: Optional industry filter (e.g. 'fintech', 'healthcare', 'saas', 'education', 'professional_services').
        min_company_size: Optional minimum company size to filter by.
    """
    leads = [
        Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None})
        for l in APOLLO_LEADS
    ]
    if industry:
        leads = [l for l in leads if l.industry.lower() == industry.lower()]
    if min_company_size:
        leads = [l for l in leads if l.company_size >= min_company_size]
    # Return as JSON string because LangGraph tools must return strings
    return json.dumps([l.model_dump() for l in leads], default=str)


@tool
def search_linkedin(industry: str = "", min_company_size: int = 0) -> str:
    """Search LinkedIn for professional contacts and potential leads.

    Args:
        industry: Optional industry filter (e.g. 'fintech', 'healthcare', 'saas', 'education', 'professional_services').
        min_company_size: Optional minimum company size to filter by.
    """
    leads = [
        Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None})
        for l in LINKEDIN_LEADS
    ]
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
    leads = [
        Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None})
        for l in WEB_SCRAPE_LEADS
    ]
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
    leads = [
        Lead(**{**l, "relevance_score": 0.0, "rejection_reason": None})
        for l in REFERRAL_LEADS
    ]
    if industry:
        leads = [l for l in leads if l.industry.lower() == industry.lower()]
    if min_company_size:
        leads = [l for l in leads if l.company_size >= min_company_size]
    return json.dumps([l.model_dump() for l in leads], default=str)


# All four tools registered for the agent
TOOLS = [search_apollo, search_linkedin, search_web, search_referrals]


# ── LangGraph state & graph ──────────────────────────────────────────────


class AgentState(TypedDict):
    """The state that flows through the LangGraph graph.
    'messages' accumulates all LLM and tool messages using operator.add
    (each node appends to the list rather than replacing it).
    """
    messages: Annotated[list[AnyMessage], operator.add]


def _build_graph():
    """Construct and compile the LangGraph StateGraph.

    The graph has two nodes:
        - "agent": calls the LLM, which either makes tool calls or returns a final answer
        - "tools": executes whatever tools the LLM requested (via ToolNode)

    The should_continue function routes:
        - If the LLM's last message has tool_calls → go to "tools" node
        - Otherwise → go to END (the LLM is done)

    After "tools" runs, it always loops back to "agent" so the LLM can
    see the tool results and decide what to do next.
    """
    # Initialize the LLM -- using Ollama running locally
    # Change this to ChatAnthropic or ChatOpenAI if you want a cloud model
    llm = ChatOllama(
        model="llama3.1",
        base_url="http://localhost:11434",
    )

    # bind_tools() tells the LLM what tools are available and how to call them
    model = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState):
        """Send the current message history to the LLM and get its response.
        The response might contain tool_calls (if the LLM wants to search)
        or just text (if it's done and summarising).
        """
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    # ToolNode automatically executes tool calls and returns ToolMessage results
    tool_node = ToolNode(TOOLS)

    def should_continue(state: AgentState):
        """Routing function: check if the LLM wants to call more tools.
        If yes → route to "tools" node. If no → route to END.
        """
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")                                       # start at agent
    graph.add_conditional_edges("agent", should_continue, ["tools", END])  # agent decides next step
    graph.add_edge("tools", "agent")                                     # after tools, go back to agent

    return graph.compile()


# ── Pipeline runner ──────────────────────────────────────────────────────

# In-memory cache so you can retrieve past runs via GET /api/leads/leads/{run_id}
_run_cache: dict[str, LeadAgentPipelineResponse] = {}

# System prompt tells the LLM how to behave and what tools it has
SYSTEM_PROMPT = (
    "You are a sales lead research agent for SimplAI AU. "
    "You have access to four lead source tools: search_apollo, search_linkedin, "
    "search_web, and search_referrals. Given the user's request, decide which "
    "sources are most relevant and call them with appropriate filters. "
    "You can call multiple tools if needed. Pass industry and min_company_size "
    "filters when the user's query implies them. "
    "After receiving tool results, provide a brief summary of what you found."
)


async def run_pipeline(
    prompt: str, min_company_size: int = 50
) -> LeadAgentPipelineResponse:
    """Execute the full lead gathering pipeline.

    Steps:
        1. Build the LangGraph graph
        2. Send the prompt to the agent (LLM reasons + calls tools)
        3. Parse tool results to extract Lead objects
        4. Log which tools were called and with what args
        5. Run all leads through the deterministic evaluator
        6. Cache and return the response

    Args:
        prompt: Natural language query from the sales manager
        min_company_size: Passed to the evaluator (not the agent -- the agent
                          may also filter by size via tool args if the LLM decides to)
    """
    start = time.monotonic()
    run_id = str(uuid.uuid4())[:8]

    graph = _build_graph()

    # Kick off the agent with system prompt + user's query
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    # Run the graph -- this does the full ReAct loop (agent → tools → agent → ...)
    result = await graph.ainvoke({"messages": messages})

    # ── Parse the results ────────────────────────────────────────────
    all_leads: list[Lead] = []
    tool_calls_log: list[ToolCallRecord] = []

    for msg in result["messages"]:
        # AI messages with tool_calls tell us what the agent decided to do
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append(
                    ToolCallRecord(
                        tool_name=tc["name"],
                        arguments=tc["args"],
                    )
                )
        # Tool messages contain the JSON results from each tool execution
        if msg.type == "tool" and msg.content:
            try:
                raw_leads = json.loads(msg.content)
                for ld in raw_leads:
                    all_leads.append(Lead(**ld))
            except (json.JSONDecodeError, TypeError):
                pass  # skip non-JSON tool outputs

    # Get the agent's final text summary (the last AI message that isn't a tool call)
    agent_summary = ""
    for msg in reversed(result["messages"]):
        if msg.type == "ai" and isinstance(msg.content, str) and msg.content:
            agent_summary = msg.content
            break

    # ── Deterministic evaluation ─────────────────────────────────────
    # This is NOT done by the LLM -- it's pure rule-based filtering/scoring
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
    """Retrieve a previously executed pipeline run from the in-memory cache."""
    return _run_cache.get(run_id)
