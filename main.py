from typing import TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.tools import tool

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from pydantic import BaseModel

from agents.wikipedia import Wikipedia

# ---------------- FastAPI app ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------- LLM ----------------
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# ---------------- Types ----------------
class QuestionRequest(BaseModel):
    messages: str
    session_id: Optional[str] = "default"

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: str

# ---------------- Tools ----------------
@tool
def search_wikipedia_by_search_query(query: str) -> str:
    """
    Search Wikipedia for the given query and return scraped HTML.
    :param query: The search query string.
    :return: The scraped HTML content of the Wikipedia page.
    """
    wiki = Wikipedia(search_query=query)
    return wiki.scrape()

@tool
def search_wikipedia_by_url(url: str) -> str:
    """
    Search Wikipedia for the given url and return scraped HTML.
    :param url: The URL of the Wikipedia page.
    :return: The scraped HTML content of the Wikipedia page.
    """
    wiki = Wikipedia(url=url)
    return wiki.scrape()

tools = [search_wikipedia_by_search_query, search_wikipedia_by_url]

llm_with_tools = llm.bind_tools(tools)

# ---------------- Graph nodes ----------------
def chatbot(state: State) -> State:
    ai_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [ai_msg]}

# ---------------- Build graph ----------------
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

checkpointer = MemorySaver() 

graph = builder.compile(checkpointer=checkpointer)

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph2.png", "wb") as f:
        f.write(png_bytes)
except Exception:
    pass

# ---------------- Routes ----------------
@app.post("/api")
def handle_question(request: QuestionRequest):
    msgs = [HumanMessage(content=request.messages)]
    config = {"configurable": {"thread_id": request.session_id}}
    result = graph.invoke({"messages": msgs}, config=config)
    print("line66", result)
    last = result["messages"][-1]
    return JSONResponse(content={"messages": last.content}, status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)