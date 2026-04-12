import os
from collections.abc import Hashable
from typing import Annotated, NotRequired

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools.convert import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pinecone import Pinecone, SearchQuery
from typing_extensions import TypedDict

AGENT_PERSONAS = {
    "customer-advocate": """You are a Customer Advocate analyst. Your role is to evaluate sales conversations
from the CUSTOMER's perspective. Focus on: how well the customer's needs were understood, whether they
felt heard, pain points that were addressed or missed, and overall customer satisfaction signals.
You can query past sales conversations for comparison using the call_vector_DB tool.
When you are satisfied with the group's analysis, include APPROVED at the end of your message.
Otherwise include DISAPPROVED.""",
    "deal-strategist": """You are a Deal Strategist analyst. Your role is to evaluate sales conversations
from a CLOSING/STRATEGY perspective. Focus on: deal progression signals, objection handling effectiveness,
pricing discussion tactics, competitive positioning, and likelihood of conversion.
You can query past sales conversations for comparison using the call_vector_DB tool.
When you are satisfied with the group's analysis, include APPROVED at the end of your message.
Otherwise include DISAPPROVED.""",
    "communications-coach": """You are a Communications Coach analyst. Your role is to evaluate sales conversations
from a COMMUNICATION QUALITY perspective. Focus on: tone, rapport building, active listening cues,
clarity of value proposition delivery, and areas where phrasing could improve.
You can query past sales conversations for comparison using the call_vector_DB tool.
When you are satisfied with the group's analysis, include APPROVED at the end of your message.
Otherwise include DISAPPROVED.""",
}

AGENTS = list(AGENT_PERSONAS.keys())


class State(TypedDict):
    messages: Annotated[
        list[AnyMessage], add_messages
    ]  # add_messages makes sure the State is append-only
    current_agent: NotRequired[str]  # tracks which agent is currently speaking


@tool
def call_vector_DB(query: str) -> str:
    """Query the Pinecone vector DB for past sales call transcripts similar to the given query."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return "Error: PINECONE_API_KEY environment variable not set"
    pc = Pinecone(api_key=api_key)
    index = pc.Index(
        host="https://sales-bot-db-demo-8zzyzqk.svc.aped-4627-b74a.pinecone.io"
    )

    filtered_results = index.search(
        namespace="salestalk",
        query=SearchQuery(
            inputs={"text": query},
            top_k=3,
        ),
        fields=["chunk_text"],
    )
    return str(filtered_results)


tools_list = [call_vector_DB]
llm = ChatOllama(model="mistral-small")
llm_with_tools = llm.bind_tools(tools_list)
tool_node = ToolNode(tools_list)


def make_agent_node(agent_name: str):
    """Create a node function for a specific agent persona."""
    system_prompt = AGENT_PERSONAS[agent_name]

    def agent_node(state: State):
        # Tag prior AI messages with agent names so the LLM sees a real discussion
        tagged_messages = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.name:
                tagged = HumanMessage(
                    content=f"[{msg.name}]: {msg.content}"
                    if msg.content
                    else f"[{msg.name}]: (used a tool)"
                )
                tagged_messages.append(tagged)
            else:
                tagged_messages.append(msg)

        messages = [
            SystemMessage(
                content=system_prompt
                + f"\n\nYour name in this discussion is [{agent_name}]. "
                "You are in a group discussion with other analysts. Build on their points, "
                "disagree where needed, and provide your own perspective."
            )
        ] + tagged_messages
        response = llm_with_tools.invoke(messages)
        # Attach agent name to the response so other agents can see who said it
        response.name = agent_name
        print(f"\n[DEBUG] Agent '{agent_name}' responded:")
        print(f"  content: {response.content[:200] if response.content else '<empty>'}")
        print(f"  tool_calls: {response.tool_calls if response.tool_calls else 'none'}")
        return {"messages": [response], "current_agent": agent_name}

    return agent_node


def route_after_agent(state: State):
    """Combined router: checks for tool calls first, then discussion flow."""
    last_message = state["messages"][-1]

    # If the LLM wants to call a tool, route to tools node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # If APPROVED, we're done
    content = last_message.content if isinstance(last_message.content, str) else ""
    if "APPROVED" in content and "DISAPPROVED" not in content:
        print("[DEBUG] Routing: APPROVED detected, ending discussion")
        return END

    # Safety limit on conversation length
    if len(state["messages"]) > 15:
        print(
            f"[DEBUG] Routing: Message limit reached ({len(state['messages'])} msgs), ending discussion"
        )
        return END

    # Route to the next agent (round-robin, skipping self)
    current: str = state.get("current_agent") or AGENTS[0]
    current_idx = AGENTS.index(current)
    next_idx = (current_idx + 1) % len(AGENTS)
    print(f"[DEBUG] Routing: '{current}' -> '{AGENTS[next_idx]}'")
    return AGENTS[next_idx]


def route_after_tools(state: State):
    """After a tool executes, return to the agent that called it."""
    return state.get("current_agent", AGENTS[0])


def build_graph():
    build = StateGraph(State)

    # Add a node per agent with its own persona
    for agent in AGENTS:
        build.add_node(agent, make_agent_node(agent))
    build.add_node("tools", tool_node)

    # Start with the first agent
    build.add_edge(START, AGENTS[0])

    # After each agent: either call tools, route to next agent, or end
    route_map: dict[Hashable, str] = {a: a for a in AGENTS}
    route_map["tools"] = "tools"
    route_map[END] = END
    for agent in AGENTS:
        build.add_conditional_edges(agent, route_after_agent, route_map)

    # After tools execute, return to whichever agent invoked them
    tools_map: dict[Hashable, str] = {a: a for a in AGENTS}
    build.add_conditional_edges("tools", route_after_tools, tools_map)

    return build.compile()


def main():
    app = build_graph()

    print("Sales Conversation Sentiment Analyzer")
    print("Paste a sales call transcription and 3 AI agents will analyze it.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("input: ")
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        # Print the final analysis
        final_content = result["messages"][-1].content
        print("\n--- Final Analysis ---")
        print(final_content if isinstance(final_content, str) else str(final_content))
        print("---\n")


main()
