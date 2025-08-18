from typing_extensions import Dict, List, TypedDict, Annotated, Optional, Any, Tuple
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
from agents.plot import ChartRenderer, GraphRenderer

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
llm = ChatOpenAI(model="gpt-5-micro", temperature=0, seed=1)

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
def search_wikipedia_by_query(query: str) -> dict[str, pd.DataFrame] | None:
    """
    Given a search query string, resolve it to a Wikipedia page and scrape its tables.
    
    :param query: Search term, e.g. "Amazon (company)".
    :return: Dict of DataFrames keyed by table headers, or None if no valid tables found.
    """
    print("Running search_wikipedia_by_query")
    wiki = Wikipedia(search_query=query)
    if not wiki.valid:
        print(f"Unable to resolve query to a valid page: {query}")
        return None
    return wiki.scrapeTable()

@tool
def search_wikipedia_by_url(url: str) -> dict[str, pd.DataFrame] | None:
    """
    Given a Wikipedia page URL, scrape and return its tables.
    
    :param url: The canonical Wikipedia URL, e.g. "https://en.wikipedia.org/wiki/Amazon_(company)".
    :return: Dict of DataFrames keyed by table headers, or None if no valid tables found.
    """
    print("Running search_wikipedia_by_url")
    wiki = Wikipedia(url=url)
    if not wiki.valid:
        print(f"Invalid or unreachable URL: {url}")
        return None
    return wiki.scrapeTable()

@tool
def bar_chart(
    data: List[Dict[str, Any]],
    x: str,
    y: str,
    title: Optional[str] = None,
    fmt: str = "png",
    return_data_uri: bool = False,
    bar_color: str = "C0",
    alpha: float = 0.9
) -> str | None:
    """
    Create a bar chart for given data.

    :param data: The data to plot (List[Dict[str, Any]]).
    :param x: The column name for the x-axis.
    :param y: The column name for the y-axis.
    :param title: The title of the plot. Defaults to "<y> by <x>" if not provided.
    :param fmt: The format of the plot (e.g. "png", "jpg", "webp"). Default is "png".
    :param return_data_uri: If True, returns a full data URI string (useful for embedding in HTML/Markdown).
    :param bar_color: The color of the bars (default is matplotlib's "C0").
    :param alpha: Transparency of the bars (0.0 = fully transparent, 1.0 = fully opaque).
    :return: The bar chart as a base64-encoded image (string).
    """
    print("running bar_chart")
    try:
        plotter = ChartRenderer(fmt=fmt)
        encoded = plotter.encode(
            pd.DataFrame(data),
            x=x,
            y=y,
            title=title,
            kind="bar",
            return_data_uri=return_data_uri,
            bar_color=bar_color,
            alpha=alpha
        )
        print("leaving bar_chart")
        return encoded
    except Exception as e:
        print("errored bar_chart", e)
        return None
    
@tool
def line_chart(
    data: List[Dict[str, Any]],
    x: str,
    y: str,
    title: Optional[str] = None,
    fmt: str = "png",
    return_data_uri: bool = False,
    line_color: str = "C0",
    alpha: float = 0.9
) -> str | None:
    """
    Create a line chart for given data.

    :param data: The data to plot (List[Dict[str, Any]]).
    :param x: The column name for the x-axis.
    :param y: The column name for the y-axis.
    :param title: The title of the plot. Defaults to "<y> over <x>" if not provided.
    :param fmt: The format of the plot (e.g. "png", "jpg", "webp"). Default is "png".
    :param return_data_uri: If True, returns a full data URI string (useful for embedding in HTML/Markdown).
    :param line_color: The color of the line.
    :param alpha: Transparency of the line (0.0 = fully transparent, 1.0 = fully opaque).
    :return: The line chart as a base64-encoded image (string).
    """
    print("running line_chart")
    try:
        plotter = ChartRenderer(fmt=fmt)
        encoded = plotter.encode(
            pd.DataFrame(data),
            x=x,
            y=y,
            title=title,
            kind="line",
            return_data_uri=return_data_uri,
            line_color=line_color,
            alpha=alpha
        )
        print("leaving line_chart")
        return encoded
    except Exception as e:
        print("errored line_chart", e)
        return None
    
@tool
def scatter_plot(
    data: List[Dict[str, Any]],
    x: str,
    y: str,
    title: Optional[str] = None,
    fmt: str = "png",
    return_data_uri: bool = False,
    color: str = "C0",
    point_size: int = 30,
    alpha: float = 0.7
) -> str | None:
    """
    Create a scatter plot for given data.

    :param data: The data to plot (list[dict], pandas.DataFrame, or {"columns": ..., "rows": ...}).
    :param x: The column name for the x-axis.
    :param y: The column name for the y-axis.
    :param title: The title of the plot. Defaults to "<x> vs <y>".
    :param fmt: The format of the plot (e.g. "png", "jpg", "webp").
    :param return_data_uri: If True, returns a full data URI string.
    :param color: Color of scatter points.
    :param point_size: Size of scatter points.
    :param alpha: Transparency of scatter points.
    :return: The scatter plot as a base64-encoded image.
    """
    print("running scatter_plot")
    try:
        plotter = ChartRenderer(fmt=fmt)
        encoded = plotter.encode(
            pd.DataFrame(data),
            x=x,
            y=y,
            title=title,
            kind="scatter",
            return_data_uri=return_data_uri,
            color=color,
            point_size=point_size,
            alpha=alpha
        )
        print("leaving scatter_plot")
        return encoded
    except Exception as e:
        print("errored scatter_plot", e)
        return None
    
@tool
def network_plot(
    data: List[Dict[str, str]],
    title: Optional[str] = None,
    fmt: str = "png",
    return_data_uri: bool = False,
    source_col: str = "source",
    target_col: str = "target"
) -> str | None:
    """
    Create a network plot from edge data and return it as a base64-encoded image.

    :param data: List of dicts with edge definitions.
                 Example: [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}]
    :param title: The title of the plot. Defaults to "Network Graph".
    :param fmt: The image format for the plot (e.g. "png", "jpg", "webp").
    :param return_data_uri: If True, return a full data URI (e.g. "data:image/png;base64,..."),
                            otherwise return just the raw base64 string.
    :param source_col: The column in the dicts representing source nodes.
    :param target_col: The column in the dicts representing target nodes.
    :return: The network plot encoded as a base64 string (or data URI).
    """
    print("running network_plot")
    try:
        # Ensure DataFrame with proper column names
        df = pd.DataFrame(data)
        if not {source_col, target_col}.issubset(df.columns):
            raise ValueError(f"Input data must contain '{source_col}' and '{target_col}' keys.")
        
        # Force all values to strings for consistent labeling
        df = df.astype(str)

        plotter = GraphRenderer(fmt=fmt)
        encoded = plotter.encode(
            df,
            title=title,
            return_data_uri=return_data_uri,
            source_col=source_col,
            target_col=target_col
        )
        print("leaving network_plot")
        return encoded
    except Exception as e:
        print("errored network_plot", e)
        return None

@tool
def scatter_plot_regression(
    data: List[Dict[str, Any]],
    x: str,
    y: str,
    title: Optional[str] = None,
    fmt: str = "png",
    return_data_uri: bool = False,
    color: str = "C0",
    line_color: str = "black",
    show_r2: bool = True
) -> str | None:
    """
    Create a scatter plot for given data with an optional regression line.

    :param data: The data to plot in the form List[Dict[str, Any]].
    :param x: The column name for the x-axis.
    :param y: The column name for the y-axis.
    :param title: The title of the plot. Defaults to "<x> vs <y>".
    :param fmt: The format of the plot (e.g. "png", "jpg", "webp").
    :param return_data_uri: If True, returns a full data URI string.
    :param color: Color for scatter points.
    :param line_color: Color for regression line.
    :param show_r2: Whether to show R² value on the plot.
    :return: The scatter plot as a base64-encoded image (string).
    """
    print("running scatter_plot_regression")
    try:
        plotter = ChartRenderer(fmt=fmt)
        encoded = plotter.encode(
            pd.DataFrame(data),
            x=x,
            y=y,
            kind="scatter",
            title=title,
            regression=True,
            show_r2=show_r2,
            return_data_uri=return_data_uri,
            color=color,
            line_color=line_color
        )
        print("leaving scatter_plot_regression")
        return encoded
    except Exception as e:
        print("errored scatter_plot_regression", e)
        return None


@tool
def duckdb_sql_query_runner(query: str) -> pd.DataFrame | str:
    """
    Execute a SQL query using DuckDB.

    This utility supports both `httpfs` and Parquet file formats. Queries may reference
    remote data sources such as S3 buckets by providing the appropriate URL.

    Example usage:
        SELECT COUNT(*) FROM read_parquet('s3://bucket-name/path/to/file.parquet');

    For date conversions, use explicit casting to ensure proper parsing:
        TRY_CAST(TRY_STRPTIME(date_of_registration, '%d-%m-%Y') AS DATE)

    :param query: The SQL query string to be executed.
    :return: A pandas DataFrame containing the query result if successful,
             otherwise an error message string.
    """
    print("Running duckdb_sql_query_runner")
    try:
        con = duckdb.connect()
        result = con.execute(query).df()
        print("Leaving duckdb_sql_query_runner")
        return result
    except Exception as e:
        print("Errored in duckdb_sql_query_runner")
        print("Error:", e)
        return str(e)

@tool
def duckdb_sql_query_runner_summary(query: str) -> str | None:
    """
    Execute a SQL query using DuckDB and return the result as a string.

    This utility supports both `httpfs` and Parquet file formats. Queries may reference
    remote data sources such as S3 buckets by providing the appropriate URL.

    Example usage:
        SELECT COUNT(*) FROM read_parquet('s3://bucket-name/path/to/file.parquet');

    For date conversions, use explicit casting to ensure proper parsing:
        TRY_CAST(TRY_STRPTIME(date_of_registration, '%d-%m-%Y') AS DATE)

    :param query: The SQL query string to be executed.
    :return: The query result as a formatted string, or None if an error occurs.
    """
    print("Running duckdb_sql_query_runner_summary")
    try:
        con = duckdb.connect()
        result = con.execute(query).df()
        print("Leaving duckdb_sql_query_runner_summary")
        return result.to_string(index=False)
    except Exception as e:
        print("Errored in duckdb_sql_query_runner_summary")
        print("Error:", e)
        return None
    

tools = [search_wikipedia_by_query, search_wikipedia_by_url, scatter_plot, scatter_plot_regression, duckdb_sql_query_runner, duckdb_sql_query_runner_summary, bar_chart, line_chart, network_plot]

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

        Use the provided *Messages* *Files* *Images* *CSVs* as context to research and answer each *Question*.
        - Instructions are provided
        - If tools are required, call them.
        - Once research is complete, return with a list of answers in the similar order as **Questions** only.
        - The final answer must contain only the final answer, with no extra text.
        - If the answer is a number, output only the number.
        - Prefer the tool output (e.g., scatterplot) unless the tool explicitly returns an error or is empty.
        - If the tool errors or returns no usable value, use the model's answer.
        - Strictly no base64 images or plots generated directly from the model.
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