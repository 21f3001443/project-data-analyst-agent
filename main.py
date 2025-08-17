from typing_extensions import Dict, List, TypedDict, Annotated, Optional, Any
import json
import base64
import mimetypes
import csv
import io
import asyncio

import pandas as pd
import duckdb

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage
from langchain_core.tools import tool

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from starlette.datastructures import UploadFile as StarletteUploadFile

from agents.wikipedia import Wikipedia
from agents.sessionstore import SessionStore
from agents.plot import ScatterPlot

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
llm = ChatOpenAI(model="gpt-5-mini", temperature=0, seed=1)

# ---------------- Session store ----------------
session_store = SessionStore(inactivity_seconds=5)  # 5 minutes

class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    instructions: list[str]
    questions: list[str]
    answers: list[str]
    files: Optional[list[str]]
    images: Optional[list[str]]
    csvs: Optional[list[str]]

# ---------------- Tools ----------------
@tool
def search_wikipedia_by_search_query(query: str) -> dict[str, pd.DataFrame] | None:
    """
    Search Wikipedia for the given query and return scraped table.
    :param query: The search query string.
    :return: The scraped table content of the Wikipedia page.
    """
    print("Running search_wikipedia_by_search_query")
    wiki = Wikipedia(search_query=query)
    normalized_data = wiki.scrapeTable()
    return normalized_data

@tool
def search_wikipedia_by_url(url: str) -> dict[str, pd.DataFrame] | None:
    """
    Search Wikipedia for the given url and return scraped table.
    :param url: The URL of the Wikipedia page.
    :return: The scraped table content of the Wikipedia page.
    """
    print("Running search_wikipedia_by_url")
    wiki = Wikipedia(url=url)
    normalized_data = wiki.scrapeTable()
    return normalized_data

@tool
def scatter_plot(data: List[Dict[str, Any]], xlabel: str, ylabel: str, title: str, fmt: str) -> str | None:
    """
    Create a scatter plot for given data.
    :param data: The data to plot in the form List[Dict[str, Any]].
    :param xlabel: The column name for the x-axis.
    :param ylabel: The column name for the y-axis.
    :param title: The title of the plot.
    :param fmt: The format of the plot (e.g. "png", "jpg", "webp").
    :return: The scatter plot as a base64-encoded image.
    """
    print("running scatter_plot")
    try:
        plotter = ScatterPlot(fmt=fmt)
        encoded = plotter.encode(data, xlabel=xlabel, ylabel=ylabel, title=title)
        print("leaving scatter_plot")
        return encoded
    except Exception as e:
        print("errored scatter_plot")
        return e

@tool
def scatter_plot_regression(data: List[Dict[str, Any]], xlabel: str, ylabel: str, title: str, fmt: str) -> str | None:
    """
    Create a scatter plot for given data with regression line.
    :param data: The data to plot in the form List[Dict[str, Any]].
    :param xlabel: The column name for the x-axis.
    :param ylabel: The column name for the y-axis.
    :param title: The title of the plot.
    :param fmt: The format of the plot (e.g. "png", "jpg", "webp").
    :return: The scatter plot as a base64-encoded image.
    """
    print("running scatter_plot_regression")
    try:
        plotter = ScatterPlot(add_regression=True, fmt=fmt)
        encoded = plotter.encode(data, xlabel=xlabel, ylabel=ylabel, title=title)
        print("leaving scatter_plot_regression")
        return encoded.split(",", 1)[1]
    except Exception as e:
        print("errored scatter_plot_regression")
        return e
    
@tool
def scatter_plot_regression(data: List[Dict[str, Any]], xlabel: str, ylabel: str, title: str, fmt: str) -> str | None:
    """
    Create a scatter plot for given data with regression line.
    :param data: The data to plot in the form List[Dict[str, Any]].
    :param xlabel: The column name for the x-axis.
    :param ylabel: The column name for the y-axis.
    :param title: The title of the plot.
    :param fmt: The format of the plot (e.g. "png", "jpg", "webp").
    :return: The scatter plot as a base64-encoded image.
    """
    print("running scatter_plot_regression")
    try:
        plotter = ScatterPlot(add_regression=True, fmt=fmt)
        encoded = plotter.encode(data, xlabel=xlabel, ylabel=ylabel, title=title)
        print("leaving scatter_plot_regression")
        return encoded.split(",", 1)[1]
    except Exception as e:
        print("errored scatter_plot_regression")
        return e


@tool
def duckdb_sql_query_runner(query: str) -> pd.DataFrame | str:
    """
    Run a SQL query against the DuckDB database.
    It supports both httpfs and parquet.
    SQL query should contain URL if the data is stored in S3/remotely. 
    Eg: SELECT COUNT(*) FROM read_parquet('s3://s3path');
    Use date format casting if required
    Eg: TRY_CAST(TRY_STRPTIME(date_of_registration, '%d-%m-%Y') AS DATE)
    :param query: The SQL query to run.
    :return: The result of the query as a DataFrame.
    """
    print("running duckdb_sql_query_runner")
    try:
        con = duckdb.connect()
        result = con.execute(query).df()
        print("Leaving duckdb_sql_query_runner")
        return result
    except Exception as e:
        print("errored duckdb_sql_query_runner")
        print("Error:", e)
        return e

@tool
def duckdb_sql_query_runner_summary(query: str) -> str | None:
    """
    Run a SQL query against the DuckDB database.
    It supports both httpfs and parquet.
    SQL query should contain URL if the data is stored in S3/remotely. 
    Eg: SELECT COUNT(*) FROM read_parquet('s3://s3path');
    Use date format casting if required
    Eg: TRY_CAST(TRY_STRPTIME(date_of_registration, '%d-%m-%Y') AS DATE)
    :param query: The SQL query to run.
    :return: The result of the query as a string.
    """
    print("running duckdb_sql_query_runner_summary")
    try:
        con = duckdb.connect()
        result = con.execute(query).df()
        print("Leaving duckdb_sql_query_runner_summary")
        return result.to_string()
    except Exception as e:
        print("Errored duckdb_sql_query_runner_summary")
        print("Error:", e)
        return 
    

tools = [search_wikipedia_by_search_query, search_wikipedia_by_url, scatter_plot, scatter_plot_regression, duckdb_sql_query_runner, duckdb_sql_query_runner_summary]

llm_with_tools = llm.bind_tools(tools)

# ---------------- Graph nodes ----------------
def split_instructions_and_questions(state: State) -> State:
    """
    Splits an incoming block of text into two categories: "instructions" and "questions".

    Behavior:
    - Extracts declarative/imperative statements (instructions).
    - Extracts interrogative queries (questions).
    - Returns a JSON object with the structure:
        {
          "instructions": [...],
          "questions": [...]
        }
    """

    system_prompt = """You are a semantic text splitter.
    Task:
    - Separate the given text into two categories:
    1. "instructions": general context, descriptions, or setup information.
    2. "questions": interrogatives (ending with "?") **and also** any requests
        that expect a generated output, calculation, chart, figure, or returned value
        (even if written as an imperative command).

    Output:
    Respond ONLY as a JSON object with two keys: "instructions" and "questions".
    """
    print("Running LLM for instruction and question splitting")
    user_prompt = f"Text:\n{state['messages']}"

    response = llm.invoke([("system", system_prompt), ("user", user_prompt)])

    try:
        # print("LLM response:", json.loads(response.content))
        # state["instructions"] = json.loads(response.content).get("instructions", [])
        # state["questions"] = json.loads(response.content).get("questions", [])
        return {"messages": [response.content], "instructions": json.loads(response.content).get("instructions", []), "questions": json.loads(response.content).get("questions", [])}
    except Exception as e:
        # fallback parsing: try to coerce into dict
        print("Error parsing LLM response:", e)
        return e
    
def research_questions(state: State) -> State:
    """
    Expects state with keys 'instructions' and 'questions'.
    """
    print("Running research_questions")
    try:
        messages = state.get("messages", [])
        instructions = state.get("instructions", [])
        questions = state.get("questions", [])
        files = state.get("files", [])
        images = state.get("images", [])
        csvs = state.get("csvs", [])

        system_prompt = """You are a helpful assistant.

        Use the provided *Messages* *Files* *Images* *CSVs* as context to reasearch and answer each *Question*.
        - instructions are provided
        - If tools are required, call them.
        - Once research is complete, return with a list of answers in the similar order as **Questions** only.
        - the final answer return only the final answer, with no extra text.
        - If the answer is a number, output only the number.
        - Prefer the tool output(Eg: scatterplot) unless the tool explicitly returns an error or is empty.
        - If the tool errors or returns no usable value, use the model's answer.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Instructions:{instructions}\nMessages:\n{messages}\nQuestions:\n{questions}\nFiles:\n{files}\nImages:\n{images}\nCSVs:\n{csvs}\n"
            )
        ]

        ai_msg = llm_with_tools.invoke(messages)
        # print("AI Message:", repr(ai_msg))
        print("Leaving research_questions")
        return {"messages": [ai_msg]}

    except Exception as e:
        print("Errored research_questions")
        print("Error answering questions:", e)
        return f"Error answering questions: {e}"

# ---------------- Build graph ----------------
builder = StateGraph(State)

builder.add_node("split_instructions_and_questions", split_instructions_and_questions)
builder.add_node("research_questions", research_questions)
# builder.add_node("fill_answers", fill_answers)
# builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "split_instructions_and_questions")
builder.add_edge("split_instructions_and_questions", "research_questions")
builder.add_conditional_edges("research_questions", tools_condition)
builder.add_edge("tools", "research_questions")
# builder.set_finish_point("fill_answers") 
# builder.set_finish_point("research_questions")

checkpointer = MemorySaver() 

graph = builder.compile(checkpointer=checkpointer)

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph2.png", "wb") as f:
        f.write(png_bytes)
except Exception:
    pass


def extract_final_answers(messages):

    try:
        return json.loads(messages.content.strip())
    except json.JSONDecodeError:
        # Fallback: put it in structured JSON instead of failing
        return messages.content.strip()

# ---------------- Routes ----------------
@app.post("/api")
async def handle_question(request: Request, session_id: str = Form("default")):

    form = await request.form()

    questions_file: UploadFile | None = None
    attachments: dict[str, UploadFile] = {}

    for field, value in form.items():
        if isinstance(value, StarletteUploadFile):
            if field.lower() == "questions.txt":   # special case
                questions_file = value
            else:
                attachments[field] = value

    # Example: Ensure "questions.txt" is always there
    if questions_file is None:
        return JSONResponse(content={"error": "questions.txt is required!"}, status_code=400)
    
    # read the question text
    questions_text = None
    if questions_file:
        raw_bytes = await questions_file.read()
        questions_text = raw_bytes.decode("utf-8", errors="ignore")
    

    file_texts = []
    image_b64s = []
    csv_data = []
    metadata = []

    for field, file in form.items():
        if not isinstance(file, StarletteUploadFile):
            continue  # skip non-file fields

        raw_bytes = await file.read()  # use await inside async endpoint
        content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
        ext = file.filename.split(".")[-1].lower() if "." in file.filename else None

        # Always record metadata
        metadata.append({
            "filename": file.filename,
            "content_type": content_type,
            "size": len(raw_bytes),
            "extension": ext
        })

        # Handle questions.txt specially
        if file.filename.lower() == "questions.txt":
            continue  # don’t process further

        # IMAGE HANDLING
        if content_type and content_type.startswith("image/"):
            print("Found image:", file.filename)
            image_b64s.append(base64.b64encode(raw_bytes).decode("utf-8"))

        # CSV HANDLING
        elif ext == "csv" or content_type in ["text/csv", "application/vnd.ms-excel"]:
            print("Found CSV:", file.filename)
            text_data = raw_bytes.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(text_data))
            rows = list(reader)
            csv_data.append({
                "filename": file.filename,
                "rows": rows
            })

        # TEXT-LIKE FILES
        elif content_type in ["text/plain", "application/json", "application/xml"]:
            print("Found text file:", file.filename)
            try:
                file_texts.append(raw_bytes.decode("utf-8"))
            except Exception:
                file_texts.append("")

        # PDF OR OTHER → fallback to base64
        else:
            file_texts.append(base64.b64encode(raw_bytes).decode("utf-8"))


    result = []
    thread_id = session_store.next_thread_id(session_id or "default")
    print("thread_id", thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    question_hm = [HumanMessage(content=questions_text)]
    file_texts_hm = [HumanMessage(content=file_texts)]
    image_b64s_hm = [HumanMessage(content=image_b64s)]
    csv_data_hm = [HumanMessage(content=csv_data)]

    # result = graph.invoke({"messages": question_hm, "files": file_texts_hm, "images": image_b64s_hm, "csvs": csv_data_hm}, config=config, debug=True)
    # result = await asyncio.to_thread(graph.invoke({"messages": question_hm, "files": file_texts_hm, "images": image_b64s_hm, "csvs": csv_data_hm}, config=config, debug=True))
    result = await asyncio.to_thread(
        graph.invoke, 
        {"messages": question_hm, "files": file_texts_hm, "images": image_b64s_hm, "csvs": csv_data_hm}, 
        config=config
    )

    final_answers = extract_final_answers(result["messages"][-1])
    # session_store.clear(session_id or "default")

    return JSONResponse(content=final_answers, status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)