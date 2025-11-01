import logging
logger = logging.getLogger(__name__)
from langgraph.messages import HumanMessage, AIMessage
from langgraph.runnables.config import RunnableConfig
from langgraph.graph.state import RuntimeState

from agent import graph


def main(thread_id: str = "book_flight", user_id: str = "demo_user"):
    """Main interaction loop for the travel agent"""

    print("Welcome to the Travel Assistant! (Type 'exit' to quit)")

    config = RunnableConfig(configurable={"thread_id": thread_id, "user_id": user_id})
    state = RuntimeState(messages=[])

    while True:
        user_input = input("\nYou (type 'quit' to quit): ")

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the Travel Assistant. Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))

        try:
            # Process user input through the graph
            for result in graph.stream(state, config=config, stream_mode="values"):
                state = RuntimeState(**result)

            logger.debug(f"# of messages after run: {len(state['messages'])}")

            # Find the most recent AI message, so we can print the response
            ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                message = ai_messages[-1].content
            else:
                logger.error("No AI messages after run")
                message = "I'm sorry, I couldn't process your request properly."
                # Add the error message to the state
                state["messages"].append(AIMessage(content=message))

            print(f"\nAssistant: {message}")

        except Exception as e:
            logger.exception(f"Error processing request: {e}")
            error_message = "I'm sorry, I encountered an error processing your request."
            print(f"\nAssistant: {error_message}")
            # Add the error message to the state
            state["messages"].append(AIMessage(content=error_message))

try:
    user_id = input("Enter a user ID: ") or "demo_user"
    thread_id = input("Enter a thread ID: ") or "demo_thread"
except Exception:
    exit()
else:
    main(thread_id, user_id)