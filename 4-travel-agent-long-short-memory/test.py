from toolkit import store_memory_tool, retrieve_memories_tool

return_status = store_memory_tool.invoke({"content": "I like flying on Delta when possible", "memory_type": "episodic"})
print(return_status)

return_status = retrieve_memories_tool.invoke({"query": "Airline preferences", "memory_type": ["episodic"]})
print(return_status)

# Test for User preferences
res = retrieve_memories_tool.invoke({"query": "Travel, activity, and dietary preferences", "memory_type": ["episodic", "semantic"]})
print(res.split("\n"))

# Total Long term memories stored
from redisvl.query import CountQuery

# count total long-term memories in Redis
long_term_memory_index.query(CountQuery())