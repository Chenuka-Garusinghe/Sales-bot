from tokenize import String
from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools.convert import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

tavily_tool = TavilySearch(max_results=3)

tools_list = [tavily_tool]
llm_with_tool = ChatOllama(model="llama3.1").bind_tools(tools_list)
tool_node = ToolNode(tools_list)


class State(TypedDict):
    messages: Annotated[
        list[AnyMessage], add_messages
    ]  # add_messages makes sure the State is append-only


def llm_chatbot(state: State):
    # Invoke the LLM with the current message history
    return {"messages": [llm_with_tool.invoke(state["messages"])]}

@tool  # get the agents to query the DB for history of similar calls (stored as trasncripts)
def call_vector_DB(qurey:str) -> String:
    pass

def build_graph():
    build = StateGraph(State)
    build.add_node("LLM", llm_chatbot)
    build.add_node("tools", tool_node)  # Node to execute tools

    build.add_edge(START, "LLM")  # Start by sending user input to the LLM

    build.add_conditional_edges(
        "LLM",
        tools_condition,  # This is a pre-built LangGraph condition: if last message has tool calls, it routes to "tools"
        # The default mapping for tools_condition is {"tools": "tools_node_name"}
    )

    build.add_edge(
        "tools", "LLM"
    )  # After tools run, send results back to the LLM for next turn

    return build.compile()


def main():
    app = build_graph()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print("Bot:", result["messages"][-1].content)


main()
