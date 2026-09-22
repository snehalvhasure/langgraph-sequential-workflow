import os
import langchain
import langgraph
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# llm_model
llm_model = ChatOpenAI(model = "gpt-4o-mini")

# STATE
class PoemState(TypedDict):
    topic : str
    poem : str
    summary : str

# Define Nodes
def generatePoem(state: PoemState):
    """Generate a poem based on the topic provided by the user."""
    topic = state["topic"]
    prompt = f"""Generate a 10 lines poem on the following topic. \n {topic}"""
    result = llm_model.invoke(prompt).content

    return {"poem" : result}

def generateSummary(state: PoemState):
    """Generate a summary on the poem provided"""
    poem = state["poem"]
    prompt2 = f"""Generate a summary about the poem provided below. \n {poem}"""
    result = llm_model.invoke(prompt2).content

    return {"summary" : result}

# GRAPH
graph = StateGraph(PoemState)

# Add NODES
graph.add_node("generatePoem", generatePoem)
graph.add_node("generateSummary", generateSummary)

# Add EDGES
graph.add_edge(START, "generatePoem")
graph.add_edge("generatePoem", "generateSummary")
graph.add_edge("generateSummary", END)

# COmpile GRAPH
poemAgent = graph.compile()
print(poemAgent)

# input topic from the user
result = poemAgent.invoke({"topic" : "INDIA"})
print(result)

print("\n\nBelow is the poem generated: ")

print(result["poem"])
print("\n\nBelow is the summary generated: ")

print(result["summary"])