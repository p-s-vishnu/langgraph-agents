from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from schema import State

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# Nodes
async def classification_node(state: State):
    '''Classify the text into one of the categories: News, Blog, Research, or Other '''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state.text))
    classification = await llm.ainvoke([message])
    return {"classification": classification.content.strip()}

async def entity_extraction_node(state: State):
    ''' Extract all the entities (Person, Organization, Location) from the text'''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state.text))
    entities = await llm.ainvoke([message])
    return {"entities": entities.content.strip().split(", ")}

async def summarization_node(state: State):
    '''Summarize the text in one short sentence'''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state.text))
    summary = await llm.ainvoke([message])
    return {"summary": summary.content.strip()}

# Graph
workflow = StateGraph(State)

workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

workflow.add_edge(START, "classification_node")
workflow.add_edge(START, "entity_extraction")
workflow.add_edge(START, "summarization")
workflow.add_edge("classification_node", END)
workflow.add_edge("entity_extraction", END)
workflow.add_edge("summarization", END)

app = workflow.compile()
