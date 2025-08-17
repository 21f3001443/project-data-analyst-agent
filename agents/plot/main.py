import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from typing import Optional, Dict, Any, Union, List

class ScatterPlot:
    """
    Render a scatter plot from:
      - pandas.DataFrame
      - {'columns': [...], 'rows': [...]} dict
      - list[dict] (records), e.g. [{'Rank':1,'Peak':1}, ...]
    and return a base64 data URI.
    """

    def __init__(
        self,
        *,
        fmt: str = "webp",
        figsize: tuple[float, float] = (4, 3),
        dpi: int = 60,
        add_regression: bool = False,
        default_title: Optional[str] = None,
    ):
        self.fmt = fmt
        self.figsize = figsize
        self.dpi = dpi
        self.add_regression = add_regression
        self.default_title = default_title

    # --- helpers -------------------------------------------------------------
    def _ensure_dataframe(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """
        Accepts:
          - DataFrame
          - {'columns': [...], 'rows': [ {...}, ... ]}
          - [ {...}, {...}, ... ]  (records)
        Returns a DataFrame with columns ordered if 'columns' provided.
        """
        # Case 1: already a DataFrame
        if isinstance(data, pd.DataFrame):
            return data

        # Case 2: list of dicts (records)
        if isinstance(data, list):
            if not data:  # empty list
                return pd.DataFrame()
            if all(isinstance(row, dict) for row in data):
                return pd.DataFrame(data)
            raise ValueError("List input must be a list of dicts (records).")

        # Case 3: dict with 'rows' / optional 'columns'
        if isinstance(data, dict) and "rows" in data:
            rows = data.get("rows", [])
            cols = data.get("columns")
            df = pd.DataFrame(rows)
            if cols:
                existing = [c for c in cols if c in df.columns]
                df = df.reindex(columns=existing + [c for c in df.columns if c not in existing])
            return df

        raise ValueError("Unsupported data type. Provide DataFrame, {'columns','rows'} dict, or list[dict].")

    _PARENS_RE = re.compile(r"^\((.*)\)$")

    def _clean_numeric_column(self, s: pd.Series) -> pd.Series:
        """Convert strings like '$2,923,706,026' or '(1,234)' to numeric."""
        if pd.api.types.is_numeric_dtype(s):
            return s
        s2 = s.astype(str).str.strip()
        s2 = s2.str.replace(self._PARENS_RE, r"-\1", regex=True)                  # (123) -> -123
        s2 = s2.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)                    # drop currency, commas, etc.
        return pd.to_numeric(s2, errors="coerce")

    # --- main API ------------------------------------------------------------
    def encode(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        xlabel: str,
        ylabel: str,
        *,
        title: Optional[str] = None,
    ) -> str:
        # 0) Normalize input -> DataFrame
        df = self._ensure_dataframe(data)

        # 1) Column checks
        missing = [c for c in (xlabel, ylabel) if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found in DataFrame: {', '.join(missing)}")

        # 2) Clean + align rows
        x = self._clean_numeric_column(df[xlabel])
        y = self._clean_numeric_column(df[ylabel])
        idx = x.notna() & y.notna()
        x = x[idx].to_numpy()
        y = y[idx].to_numpy()
        if x.size == 0:
            raise ValueError("No valid numeric data to plot after cleaning.")

        # 3) Plot
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.scatter(x, y)

        if self.add_regression and x.size >= 2:
            a, b = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 200)
            ys = a * xs + b
            plt.plot(xs, ys, linestyle=":", linewidth=1.2)  # no explicit color

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title if title is not None else self.default_title)
        plt.tight_layout()

        # 4) Encode -> data URI
        buf = io.BytesIO()
        fig.savefig(buf, format=self.fmt)
        plt.close(fig)

        data_uri = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/{self.fmt};base64,{data_uri}"