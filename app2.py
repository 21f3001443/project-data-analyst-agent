from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    target_currency: Literal["INR", "EUR"]
    total: float

def calc_total(state: PortfolioState) -> PortfolioState:
    """
    Calculate the total value in USD.
    """
    state['total_usd'] = state['amount_usd'] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    """
    Convert the total value to INR.
    """
    state['total'] = state['total_usd'] * 85
    return state

def convert_to_eur(state: PortfolioState) -> PortfolioState:
    """
    Convert the total value to EUR.
    """
    state['total'] = state['total_usd'] * 0.9
    return state

def choose_conversion(state: PortfolioState) -> str:
    """
    Choose the conversion function based on the target currency.
    """
    return state['target_currency']

builder = StateGraph(PortfolioState)

builder.add_node("calc_total", calc_total)
builder.add_node("convert_to_inr", convert_to_inr)
builder.add_node("convert_to_eur", convert_to_eur)

builder.add_edge(START, "calc_total")
builder.add_conditional_edges(
    "calc_total",
    choose_conversion,
    {
        "INR": "convert_to_inr",
        "EUR": "convert_to_eur"
    }
)
builder.add_edge("convert_to_inr", END)
builder.add_edge("convert_to_eur", END)

graph = builder.compile()

png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph2.png", "wb") as f:
    f.write(png_bytes)
