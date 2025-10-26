from langgraph.graph import START, END, StateGraph

from utils.nodes import (
    generate_joke,
    check_joke_quality,
    route_joke_quality,
    improve_joke,
    finalize_joke,
    compose_poem,
    write_story,
    aggregate_answers,
)
from utils.state import State



def _build_workflow() -> StateGraph:
    workflow = StateGraph(State)

    # Joke nodes
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("check_joke_quality", check_joke_quality)
    workflow.add_node("improve_joke", improve_joke)
    workflow.add_node("finalize_joke", finalize_joke)
    # Poem node
    workflow.add_node("compose_poem", compose_poem)
    # Story node
    workflow.add_node("write_story", write_story)
    workflow.add_node("aggregate_answers", aggregate_answers, defer=True)

    workflow.add_edge(START, "generate_joke")
    workflow.add_edge("generate_joke", "check_joke_quality")
    workflow.add_conditional_edges(
        "check_joke_quality",
        route_joke_quality,
        {"passed": "finalize_joke", "retry": "improve_joke", "give_up": "finalize_joke"},
    )
    workflow.add_edge("improve_joke", "check_joke_quality")
    workflow.add_edge("finalize_joke", "aggregate_answers")
    
    workflow.add_edge(START, "compose_poem")
    workflow.add_edge("compose_poem", "aggregate_answers")

    workflow.add_edge(START, "write_story")
    workflow.add_edge("write_story", "aggregate_answers")

    workflow.add_edge("aggregate_answers", END)

    return workflow


aggregator_app = _build_workflow().compile()
