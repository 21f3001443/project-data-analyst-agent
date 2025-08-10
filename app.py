from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_inr: float

def calc_total(state: PortfolioState) -> PortfolioState:
    """
    Calculate the total value in USD.
    """
    return state['amount_usd'] * 1.08

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    """
    Convert the total value to INR.
    """
    state['total_inr'] = state['total_usd'] * 85
    return state



builder = StateGraph(PortfolioState)

builder.add_node("calc_total", calc_total)
builder.add_node("convert_to_inr", convert_to_inr)

builder.add_edge(START, "calc_total")
builder.add_edge("calc_total", "convert_to_inr")
builder.add_edge("convert_to_inr", END)

graph = builder.compile()

png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
