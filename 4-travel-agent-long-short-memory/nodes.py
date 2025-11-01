import logging
logger = logging.getLogger(__name__)

from redis_utils import redis_saver
from toolkit import store_memory_tool, retrieve_memories_tool
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
)
from langgraph.graph.message import MessagesState


tools = [store_memory_tool, retrieve_memories_tool]
llm = ChatOpenAI(model="gpt-4.1", temperature=0.7).bind_tools(tools)
summarizer = ChatOpenAI(model="gpt-4.1", temperature=0.3)
# The number of messages after which we'll summarize the conversation.
MESSAGE_SUMMARIZATION_THRESHOLD = 6


travel_agent = create_react_agent(
    model=llm,
    tools=tools,               # Long-term memory: provided as a set of custom tools
    checkpointer=redis_saver,  # Short-term memory: the conversation history
    prompt=SystemMessage(
        content="""
        You are a travel assistant helping users plan their trips. You remember user preferences
        and provide personalized recommendations based on past interactions.

        You have access to the following types of memory:
        1. Short-term memory: The current conversation thread
        2. Long-term memory:
           - Episodic: User preferences and past trip experiences (e.g., "User prefers window seats")
           - Semantic: General knowledge about travel destinations and requirements

        Your procedural knowledge (how to search, book flights, etc.) is built into your tools and prompts.

        Always be helpful, personal, and context-aware in your responses.
        """
    ),
)


class RuntimeState(MessagesState):
    """Runtime state for the travel agent."""
    pass


def respond_to_user(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    """Step 1: Entrypoint"""
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        logger.warning("No HumanMessage found in state")
        return state

    try:
        result = travel_agent.invoke({"messages": state["messages"]}, config=config)
        agent_message = result["messages"][-1]
        state["messages"].append(agent_message)
    except Exception as e:
        logger.error(f"Error invoking travel agent: {e}")
        agent_message = AIMessage(
            content="I'm sorry, I encountered an error processing your request."
        )
        state["messages"].append(agent_message)
    return state


def execute_tools(state: RuntimeState, config: RunnableConfig) -> RuntimeState:
    """Step 2: Execute tools specified in the latest AIMessage and append ToolMessages."""
    messages = state["messages"]
    latest_ai_message = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
        None
    )

    if not latest_ai_message:
        return state  # No tool calls to process

    tool_messages = []
    for tool_call in latest_ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            continue

        try:
            result = tool.invoke(tool_args, config=config)
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(tool_message)
        except Exception as e:
            error_message = ToolMessage(
                content=f"Error executing tool '{tool_name}': {str(e)}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_messages.append(error_message)
    messages.extend(tool_messages)
    state["messages"] = messages
    return state

def summarize_conversation(
    state: RuntimeState, config: RunnableConfig
) -> RuntimeState:
    """
    Summarize a list of messages into a concise summary to reduce context length
    while preserving important information.
    """
    messages = state["messages"]
    current_message_count = len(messages)
    if current_message_count < MESSAGE_SUMMARIZATION_THRESHOLD:
        logger.debug(f"Not summarizing conversation: {current_message_count}")
        return state

    system_prompt = """
    You are a conversation summarizer. Create a concise summary of the previous
    conversation between a user and a travel assistant.

    The summary should:
    1. Highlight key topics, preferences, and decisions
    2. Include any specific trip details (destinations, dates, preferences)
    3. Note any outstanding questions or topics that need follow-up
    4. Be concise but informative

    Format your summary as a brief narrative paragraph.
    """

    message_content = "\n".join(
        [
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages
        ]
    )

    # Invoke the summarizer
    summary_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Please summarize this conversation:\n\n{message_content}"
        ),
    ]
    summary_response = summarizer.invoke(summary_messages)

    logger.info(f"Summarized {len(messages)} messages into a conversation summary")
    summary_message = SystemMessage(
        content=f"""
        Summary of the conversation so far:

        {summary_response.content}

        Please continue the conversation based on this summary and the recent messages.
        """
    )
    remove_messages = [
        RemoveMessage(id=msg.id) for msg in messages if msg.id is not None
    ]
    state["messages"] = [  # type: ignore
        *remove_messages,
        summary_message,
        state["messages"][-1],
    ]
    return state.copy()

def decide_next_step(state):
    latest_ai_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if latest_ai_message and latest_ai_message.tool_calls:
        return "execute_tools"
    return "summarize_conversation"