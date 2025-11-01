from nodes import (
    respond_to_user,
    execute_tools,
    summarize_conversation,
    decide_next_step,

)
from redis_utils import redis_saver
from langgraph.graph import StateGraph, END

workflow = StateGraph(RuntimeState)

# Add nodes to the graph
workflow.add_node("agent", respond_to_user)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("summarize_conversation", summarize_conversation)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    decide_next_step,
    {"execute_tools": "execute_tools", "summarize_conversation": "summarize_conversation"},
)
workflow.add_edge("execute_tools", "agent")
workflow.add_edge("summarize_conversation", END)
graph = workflow.compile(checkpointer=redis_saver())