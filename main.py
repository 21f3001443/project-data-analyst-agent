from typing import TypedDict, Literal
import os

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import OpenAI

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import httpx

from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),http_client=httpx.Client(verify=False))
llm = ChatOpenAI(model="gpt-4o", temperature=0)

class QuestionRequest(BaseModel):
    messages: str

class State(TypedDict):
    messages: list

def chatbot(state: State) -> State:
    response = llm.invoke([HumanMessage(content=state["messages"])])
    print(response)
    return response.content

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph2.png", "wb") as f:
    f.write(png_bytes)

@app.post("/api")
def handle_question(request: QuestionRequest):
    return JSONResponse(content=graph.invoke({"messages": [request.messages]}), status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)