import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from typing import Optional, Dict, Any, Union, List, Literal, Tuple
import networkx as nx
from langchain_core.documents.base import Blob


class ChartRenderer:
    """
    Render scatter, bar, or line charts from:
      - pandas.DataFrame
      - {'columns': [...], 'rows': [...]}
      - list[dict] (records)
    Returns a base64-encoded image, either raw string or data URI.
    """

    def __init__(
        self,
        *,
        fmt: str = "webp",
        figsize: tuple[float, float] = (3, 2),
        dpi: int = 90,
        default_title: Optional[str] = None,
        grid: bool = True,
    ):
        self.fmt = fmt
        self.figsize = figsize
        self.dpi = dpi
        self.default_title = default_title
        self.grid = grid

    # --- helpers -------------------------------------------------------------
    def _ensure_dataframe(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Normalize input into DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            if all(isinstance(row, dict) for row in data):
                return pd.DataFrame(data)
            raise ValueError("List input must be list[dict].")

        if isinstance(data, dict) and "rows" in data:
            rows = data.get("rows", [])
            cols = data.get("columns")
            df = pd.DataFrame(rows)
            if cols:
                existing = [c for c in cols if c in df.columns]
                df = df.reindex(columns=existing + [c for c in df.columns if c not in existing])
            return df

        raise ValueError("Unsupported data type.")

    _PARENS_RE = re.compile(r"^\((.*)\)$")

    def _clean_numeric_column(self, s: pd.Series) -> pd.Series:
        """Convert strings like '$2,923' or '(123)' to numeric."""
        if pd.api.types.is_numeric_dtype(s):
            return s
        s2 = s.astype(str).str.strip()
        s2 = s2.str.replace(self._PARENS_RE, r"-\1", regex=True)     # (123) -> -123
        s2 = s2.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)       # drop non-numeric
        return pd.to_numeric(s2, errors="coerce")

    # def _encode_plot(self, fig) -> str:
    #     """Encode Matplotlib fig to base64 string."""
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format=self.fmt)
    #     plt.close(fig)
    #     buf.seek(0)
    #     data_uri = base64.b64encode(buf.getvalue()).decode("ascii")
    #     buf.close()
    #     return data_uri


    def _encode_plot(self, fig, return_data_uri: bool = False):
        """Encode Matplotlib fig to Blob or base64 string based on flag."""
        buf = io.BytesIO()
        fig.savefig(buf, format=self.fmt)
        plt.close(fig)
        buf.seek(0)

        if return_data_uri:
            # old behavior — return base64 string / data URI
            data_uri = base64.b64encode(buf.getvalue()).decode("ascii")
            buf.close()
            return f"data:image/{self.fmt};base64,{data_uri}"
        else:
            # new behavior — return LangGraph Blob
            return Blob.from_data(
                buf.getvalue(),
                mime_type=f"image/{self.fmt}",
                path=f"plot.{self.fmt}"
            )

    # --- main API ------------------------------------------------------------
    def encode(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        x: str,
        y: str,
        *,
        kind: Literal["scatter", "bar", "line"] = "scatter",
        title: Optional[str] = None,
        return_data_uri: bool = False,
        point_size: int = 30,
        alpha: float = 0.7,
        color: str = "C0",
        line_color: str = "black",
        bar_color: str = "C1",
        regression: bool = False,
        show_r2: bool = False,
    ) -> Blob | str:
        """
        Create a chart and return base64 string.

        kind: 'scatter' | 'bar' | 'line'
        """

        # 0) Normalize
        df = self._ensure_dataframe(data)

        # 1) Column checks
        missing = [c for c in (x, y) if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found: {', '.join(missing)}")

        # 2) Clean + align
        if kind == "bar" or kind == "line":
            # allow categorical x
            X = df[x].astype(str)
            Y = self._clean_numeric_column(df[y])
            idx = Y.notna()
            X, Y = X[idx], Y[idx]
            if Y.empty:
                raise ValueError("No valid numeric Y data to plot.")
        else:
            # scatter / line → both numeric
            X = self._clean_numeric_column(df[x])
            Y = self._clean_numeric_column(df[y])
            idx = X.notna() & Y.notna()
            X, Y = X[idx], Y[idx]
            if X.empty:
                raise ValueError("No valid numeric data to plot.")

        # 3) Plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if kind == "scatter":
            ax.scatter(X, Y, s=point_size, alpha=alpha, color=color)
            if regression and len(X) >= 2:
                a, b = np.polyfit(X, Y, 1)
                xs = np.linspace(float(X.min()), float(X.max()), 200)
                ys = a * xs + b
                ax.plot(xs, ys, linestyle=":", color=line_color, linewidth=1.2)
                if show_r2:
                    y_pred = a * X + b
                    ss_res = np.sum((Y - y_pred) ** 2)
                    ss_tot = np.sum((Y - Y.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot
                    ax.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes,
                            ha="left", va="top", fontsize=9,
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        elif kind == "bar":
            ax.bar(X, Y, color=bar_color, alpha=alpha)

        elif kind == "line":
            ax.plot(X, Y, marker="o", linestyle="-", color=color, alpha=alpha)

        else:
            raise ValueError(f"Unsupported chart kind: {kind}")

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title if title else self.default_title or f"{x} vs {y}")
        if self.grid:
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()

        # 4) Encode
        img_str = self._encode_plot(fig)
        return f"data:image/{self.fmt};base64,{img_str}" if return_data_uri else img_str
    

class GraphRenderer:
    """
    Render a network (graph) with labeled nodes and edges.
    Accepts either:
      - list of edges (tuples), e.g. [("A", "B"), ("B", "C")]
      - pandas.DataFrame with two columns: source and target
    Returns a base64-encoded image, either raw string or data URI.
    """

    def __init__(
        self,
        *,
        fmt: str = "png",
        figsize: Tuple[float, float] = (3, 3),
        dpi: int = 100,
        default_title: Optional[str] = "Network Graph",
        layout: Literal["spring", "circular", "kamada_kawai", "shell"] = "spring",
        node_color: str = "lightblue",
        edge_color: str = "gray",
        node_size: int = 1200,
        font_size: int = 12,
        directed: bool = False,
    ):
        self.fmt = fmt
        self.figsize = figsize
        self.dpi = dpi
        self.default_title = default_title
        self.layout = layout
        self.node_color = node_color
        self.edge_color = edge_color
        self.node_size = node_size
        self.font_size = font_size
        self.directed = directed

    # --- helpers -------------------------------------------------------------
    # def _encode_plot(self, fig) -> str:
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format=self.fmt)
    #     plt.close(fig)
    #     buf.seek(0)
    #     img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    #     buf.close()
    #     return img_str

    def _encode_plot(self, fig, return_data_uri: bool = False):
        """Encode Matplotlib fig to Blob or base64 string based on flag."""
        buf = io.BytesIO()
        fig.savefig(buf, format=self.fmt)
        plt.close(fig)
        buf.seek(0)

        if return_data_uri:
            # old behavior — return base64 string / data URI
            data_uri = base64.b64encode(buf.getvalue()).decode("ascii")
            buf.close()
            return f"data:image/{self.fmt};base64,{data_uri}"
        else:
            # new behavior — return LangGraph Blob
            return Blob.from_data(
                buf.getvalue(),
                mime_type=f"image/{self.fmt}",
                path=f"plot.{self.fmt}"
            )

    def _get_layout(self, G):
        if self.layout == "spring":
            return nx.spring_layout(G, seed=42)
        elif self.layout == "circular":
            return nx.circular_layout(G)
        elif self.layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        elif self.layout == "shell":
            return nx.shell_layout(G)
        else:
            return nx.spring_layout(G, seed=42)

    # --- main API ------------------------------------------------------------
    def encode(
        self,
        edges: Union[List[Tuple[str, str]], pd.DataFrame],
        *,
        title: Optional[str] = None,
        return_data_uri: bool = False,
        source_col: str = "source",
        target_col: str = "target",
    ) -> Blob | str:
        """
        Draw the network with nodes labeled and edges shown.
        Accepts list of tuples or DataFrame (with source and target columns).
        Return as base64 string or data URI.
        """

        # 1) Normalize edges
        if isinstance(edges, pd.DataFrame):
            if not {source_col, target_col}.issubset(edges.columns):
                raise ValueError(f"DataFrame must have '{source_col}' and '{target_col}' columns.")
            edges = edges.astype(str)
            edge_list = list(edges[[source_col, target_col]].itertuples(index=False, name=None))
        else:
            edge_list = edges

        # 2) Build graph
        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_edges_from(edge_list)

        # 3) Layout
        pos = self._get_layout(G)

        # 4) Draw
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        nx.draw(
            G, pos, with_labels=True,
            node_color=self.node_color,
            edge_color=self.edge_color,
            node_size=self.node_size,
            font_size=self.font_size,
            ax=ax
        )

        ax.set_title(title or self.default_title)
        plt.tight_layout()

        # 5) Encode
        img_str = self._encode_plot(fig)
        return f"data:image/{self.fmt};base64,{img_str}" if return_data_uri else img_str
    
# class ChartRenderer:
#     """
#     Render scatter, bar, or line charts from:
#       - pandas.DataFrame
#       - {'columns': [...], 'rows': [...]}
#       - list[dict] (records)
#     Returns a base64-encoded image, either raw string or data URI.
#     """

#     def __init__(
#         self,
#         *,
#         fmt: str = "webp",
#         figsize: tuple[float, float] = (3, 2),
#         dpi: int = 90,
#         default_title: Optional[str] = None,
#         grid: bool = True,
#     ):
#         self.fmt = fmt
#         self.figsize = figsize
#         self.dpi = dpi
#         self.default_title = default_title
#         self.grid = grid

#     # --- helpers -------------------------------------------------------------
#     def _ensure_dataframe(
#         self,
#         data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
#     ) -> pd.DataFrame:
#         """Normalize input into DataFrame."""
#         if isinstance(data, pd.DataFrame):
#             return data

#         if isinstance(data, list):
#             if not data:
#                 return pd.DataFrame()
#             if all(isinstance(row, dict) for row in data):
#                 return pd.DataFrame(data)
#             raise ValueError("List input must be list[dict].")

#         if isinstance(data, dict) and "rows" in data:
#             rows = data.get("rows", [])
#             cols = data.get("columns")
#             df = pd.DataFrame(rows)
#             if cols:
#                 existing = [c for c in cols if c in df.columns]
#                 df = df.reindex(columns=existing + [c for c in df.columns if c not in existing])
#             return df

#         raise ValueError("Unsupported data type.")

#     _PARENS_RE = re.compile(r"^\((.*)\)$")

#     def _clean_numeric_column(self, s: pd.Series) -> pd.Series:
#         """Convert strings like '$2,923' or '(123)' to numeric."""
#         if pd.api.types.is_numeric_dtype(s):
#             return s
#         s2 = s.astype(str).str.strip()
#         s2 = s2.str.replace(self._PARENS_RE, r"-\1", regex=True)     # (123) -> -123
#         s2 = s2.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)       # drop non-numeric
#         return pd.to_numeric(s2, errors="coerce")

#     def _encode_plot(self, fig) -> str:
#         """Encode Matplotlib fig to base64 string."""
#         buf = io.BytesIO()
#         fig.savefig(buf, format=self.fmt)
#         plt.close(fig)
#         buf.seek(0)
#         data_uri = base64.b64encode(buf.getvalue()).decode("ascii")
#         buf.close()
#         return data_uri

#     # --- main API ------------------------------------------------------------
#     def encode(
#         self,
#         data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
#         x: str,
#         y: str,
#         *,
#         kind: Literal["scatter", "bar", "line"] = "scatter",
#         title: Optional[str] = None,
#         return_data_uri: bool = False,
#         point_size: int = 30,
#         alpha: float = 0.7,
#         color: str = "C0",
#         line_color: str = "black",
#         bar_color: str = "C1",
#         regression: bool = False,
#         show_r2: bool = False,
#     ) -> str:
#         """
#         Create a chart and return base64 string.

#         kind: 'scatter' | 'bar' | 'line'
#         """

#         # 0) Normalize
#         df = self._ensure_dataframe(data)

#         # 1) Column checks
#         missing = [c for c in (x, y) if c not in df.columns]
#         if missing:
#             raise ValueError(f"Column(s) not found: {', '.join(missing)}")

#         # 2) Clean + align
#         if kind == "bar":
#             # allow categorical x
#             X = df[x].astype(str)
#             Y = self._clean_numeric_column(df[y])
#             idx = Y.notna()
#             X, Y = X[idx], Y[idx]
#             if Y.empty:
#                 raise ValueError("No valid numeric Y data to plot.")
#         else:
#             # scatter / line → both numeric
#             X = self._clean_numeric_column(df[x])
#             Y = self._clean_numeric_column(df[y])
#             idx = X.notna() & Y.notna()
#             X, Y = X[idx], Y[idx]
#             if X.empty:
#                 raise ValueError("No valid numeric data to plot.")

#         # 3) Plot
#         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

#         if kind == "scatter":
#             ax.scatter(X, Y, s=point_size, alpha=alpha, color=color)
#             if regression and len(X) >= 2:
#                 a, b = np.polyfit(X, Y, 1)
#                 xs = np.linspace(float(X.min()), float(X.max()), 200)
#                 ys = a * xs + b
#                 ax.plot(xs, ys, linestyle=":", color=line_color, linewidth=1.2)
#                 if show_r2:
#                     y_pred = a * X + b
#                     ss_res = np.sum((Y - y_pred) ** 2)
#                     ss_tot = np.sum((Y - Y.mean()) ** 2)
#                     r2 = 1 - ss_res / ss_tot
#                     ax.text(0.05, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes,
#                             ha="left", va="top", fontsize=9,
#                             bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

#         elif kind == "bar":
#             ax.bar(X, Y, color=bar_color, alpha=alpha)

#         elif kind == "line":
#             ax.plot(X, Y, marker="o", linestyle="-", color=color, alpha=alpha)

#         else:
#             raise ValueError(f"Unsupported chart kind: {kind}")

#         ax.set_xlabel(x)
#         ax.set_ylabel(y)
#         ax.set_title(title if title else self.default_title or f"{x} vs {y}")
#         if self.grid:
#             ax.grid(True, linestyle="--", alpha=0.6)

#         plt.tight_layout()

#         # 4) Encode
#         img_str = self._encode_plot(fig)
#         return f"data:image/{self.fmt};base64,{img_str}" if return_data_uri else img_str
    

# class GraphRenderer:
#     """
#     Render a network (graph) with labeled nodes and edges.
#     Accepts either:
#       - list of edges (tuples), e.g. [("A", "B"), ("B", "C")]
#       - pandas.DataFrame with two columns: source and target
#     Returns a base64-encoded image, either raw string or data URI.
#     """

#     def __init__(
#         self,
#         *,
#         fmt: str = "png",
#         figsize: Tuple[float, float] = (3, 3),
#         dpi: int = 100,
#         default_title: Optional[str] = "Network Graph",
#         layout: Literal["spring", "circular", "kamada_kawai", "shell"] = "spring",
#         node_color: str = "lightblue",
#         edge_color: str = "gray",
#         node_size: int = 1200,
#         font_size: int = 12,
#         directed: bool = False,
#     ):
#         self.fmt = fmt
#         self.figsize = figsize
#         self.dpi = dpi
#         self.default_title = default_title
#         self.layout = layout
#         self.node_color = node_color
#         self.edge_color = edge_color
#         self.node_size = node_size
#         self.font_size = font_size
#         self.directed = directed

#     # --- helpers -------------------------------------------------------------
#     def _encode_plot(self, fig) -> str:
#         buf = io.BytesIO()
#         fig.savefig(buf, format=self.fmt)
#         plt.close(fig)
#         buf.seek(0)
#         img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
#         buf.close()
#         return img_str

#     def _get_layout(self, G):
#         if self.layout == "spring":
#             return nx.spring_layout(G, seed=42)
#         elif self.layout == "circular":
#             return nx.circular_layout(G)
#         elif self.layout == "kamada_kawai":
#             return nx.kamada_kawai_layout(G)
#         elif self.layout == "shell":
#             return nx.shell_layout(G)
#         else:
#             return nx.spring_layout(G, seed=42)

#     # --- main API ------------------------------------------------------------
#     def encode(
#         self,
#         edges: Union[List[Tuple[str, str]], pd.DataFrame],
#         *,
#         title: Optional[str] = None,
#         return_data_uri: bool = False,
#         source_col: str = "source",
#         target_col: str = "target",
#     ) -> str:
#         """
#         Draw the network with nodes labeled and edges shown.
#         Accepts list of tuples or DataFrame (with source and target columns).
#         Return as base64 string or data URI.
#         """

#         # 1) Normalize edges
#         if isinstance(edges, pd.DataFrame):
#             if not {source_col, target_col}.issubset(edges.columns):
#                 raise ValueError(f"DataFrame must have '{source_col}' and '{target_col}' columns.")
#             edges = edges.astype(str)
#             edge_list = list(edges[[source_col, target_col]].itertuples(index=False, name=None))
#         else:
#             edge_list = edges

#         # 2) Build graph
#         G = nx.DiGraph() if self.directed else nx.Graph()
#         G.add_edges_from(edge_list)

#         # 3) Layout
#         pos = self._get_layout(G)

#         # 4) Draw
#         fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
#         nx.draw(
#             G, pos, with_labels=True,
#             node_color=self.node_color,
#             edge_color=self.edge_color,
#             node_size=self.node_size,
#             font_size=self.font_size,
#             ax=ax
#         )

#         ax.set_title(title or self.default_title)
#         plt.tight_layout()

#         # 5) Encode
#         img_str = self._encode_plot(fig)
#         return f"data:image/{self.fmt};base64,{img_str}" if return_data_uri else img_str