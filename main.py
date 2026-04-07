from random import random
from tokenize import String
from typing import Annotated
import os

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools.convert import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone
from typing_extensions import TypedDict

SYSTEM_CONDITIONS = """
If you the agent are satisfied with the outcome of the discussion so far about the sentiment of the
users input (a transcription of the call he had) appened to the string: APPROVED else append: DISAPPROVE"""


class State(TypedDict):
    messages: Annotated[
        list[AnyMessage], add_messages
    ]  # add_messages makes sure the State is append-only


@tool  # get the agents to query the pinconeDB for history of similar calls (stored as trasncripts)
def call_vector_DB(query: str) -> String:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(host="sales-bot-db-demo")
    
    filtered_results = index.search(
        namespace="salestalk", 
        query={
            "inputs": {"text": query}, 
            "top_k": 3,
        },
        fields=["chunk_text"]
    )
    return filtered_results


def llm_chatbot(state: State):
    # Invoke the LLM with the current message history
    return {"messages": [llm_with_tool.invoke(state["messages"])]}


# getting the agents to talk to each other through `router`
# due to non-determinstic nature, router will have a exit after some itterations
def router(state: State):
    messages = state["messages"]
    prev_message = messages[-1].content

    if "APPROVED" in prev_message:
        return "FINISHED"
    if len(messages) > 15:
        return "FINISHED"
    return "drafter"


tools_list = [call_vector_DB]
llm_with_tool = ChatOllama(model="llama3.1").bind_tools(tools_list)
tool_node = ToolNode(tools_list)


def build_graph():
    build = StateGraph(State)
    agents = [
        "LLM-1-Customer-advocate",
        "LLM-2-Deal-strategist",
        "LLM-3-Communications-coach",
    ]
    for agent in agents:
        build.add_node(agent, llm_chatbot)
    build.add_node("tools", tool_node)  # Node to execute tools

    build.add_edge(
        START, random.choice(agents)
    )  # Start by sending user input to the LLM

    for agent in agents:
        build.add_edge("tools", agent)

    for agent in agents:
        random_agent = random.choice(agents)
        build.add_conditional_edges(
            agent,
            tools_condition,  # This is a pre-built LangGraph condition: if last message has tool calls, it routes to "tools"
            # The default mapping for tools_condition is {"tools": "tools_node_name"}
        )
        build.add_conditional_edges(
            agent,
            router,
            {
                agent: random_agent if agent != random_agent else "" # talking agents with each other but not self,
                "FINISHED": END
            })

    # connect agents with each other
    for i in range(0, len(agents)):
        for j in range(i + 1, len(agents)):
            build.add_edge(agents[i], agents[j])


    return build.compile()


def main():
    app = build_graph()

    while True:
        user_input = input("You: " + SYSTEM_CONDITIONS)
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print("Bot:", result["messages"][-1].content)


main()
