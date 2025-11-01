# Travel Agent

An agentic workflow (ReAct travel agent is a node in it).

- Dual-Memory Architecture: Short-term conversation state + long-term persistent knowledge with LangGraph and Redis
- Vector-Powered Memory: Semantic search
- Smart Deduplication: Prevents storing similar memories multiple times
- Tool-Based Memory Management: LLM controls when to store/retrieve memories
- Conversation Summarization: Automatic context window management

## Memory component - Dual-Memory Architecture
1. Short-term memory: LangGraph Redis [checkpointer](https://github.com/redis-developer/langgraph-redis).
2. Long-term memory: VectorDB

### Local Redis setup

```shell
%%sh
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update  > /dev/null 2>&1
sudo apt-get install redis-stack-server  > /dev/null 2>&1
redis-stack-server --daemonize yes
```

Once redis is created, RVL cli

```shell
rvl index info -i agent_memories
```

## Tool calling -  Tool-Based Memory Management

1. Store memory
1. Load episodic or semantic memories

**Tool-based memory management:**

- LLM decides when to store/retrieve memories
- Fewer Redis calls but may miss some context
- Adds some latency due to LLM decision-making

Alternatively, you can always manually manage memories in your workflows.

**Manual memory management:**

- More Redis calls but faster response times
- Extracts more memories, providing richer context
- Higher token usage due to more context

