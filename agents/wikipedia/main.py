import wikipedia
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
import re
import numpy as np
from typing import Any

class Wikipedia:
    """
    Create with either search_query or url.
    If only search_query is provided, find_url() runs automatically.
    Attributes:
      - search_query: str | None
      - url: str | None
      - valid: bool
      - title: str | None (resolved page title when found)
    """

    # ---------------------------
    # Regex helpers (reused)
    # ---------------------------
    _NUM_RE = re.compile(r"-?\d[\d,\.]*")
    _MONEY_RE = re.compile(r"(\d[\d,\.]*)\s*(million|billion)?", re.I)

    def __init__(self, search_query: str | None = None, url: str | None = None, *, language: str = "en"):
        self.search_query = search_query
        self.url = url
        self.valid = False
        self.title = None
        wikipedia.set_lang(language)

        if not self.url and self.search_query:
            self.find_url()
        elif self.url:
            self.check_url()

    # ---------------------------
    # URL discovery/validation
    # ---------------------------
    def find_url(self) -> None:
        """Resolve self.search_query to a canonical Wikipedia page URL, setting url/title/valid."""
        q = (self.search_query or "").strip()
        if not q:
            self.url = None
            self.title = None
            self.valid = False
            return

        try:
            results = wikipedia.search(q) or []
        except Exception:
            results = []

        candidates = []
        if results:
            candidates.append(("search_top", results[0]))
        candidates.append(("autosuggest_query", q))

        for source, title in candidates:
            try:
                page = wikipedia.page(title, auto_suggest=(source == "autosuggest_query"), redirect=True)
                self.url = page.url
                self.title = page.title
                self.valid = True
                return
            except wikipedia.DisambiguationError as e:
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
                        self.url = page.url
                        self.title = page.title
                        self.valid = True
                        return
                    except Exception:
                        pass
            except wikipedia.PageError:
                continue
            except Exception:
                continue

        self.url = None
        self.title = None
        self.valid = False

    def check_url(self) -> None:
        """Validate that self.url is a reachable Wikipedia page."""
        if not self.url or not self.url.startswith("https://en.wikipedia.org/wiki/"):
            self.valid = False
            return

        try:
            resp = requests.head(self.url, allow_redirects=True, timeout=5)
            if resp.status_code == 200 and "wikipedia.org" in resp.url:
                try:
                    page_title = self.url.split("/wiki/")[-1].replace("_", " ")
                    page = wikipedia.page(page_title, auto_suggest=False, redirect=True)
                    self.title = page.title
                except Exception:
                    self.title = None
                self.valid = True
            else:
                self.valid = False
        except Exception:
            self.valid = False

    # ---------------------------
    # Fetch & HTML cleanup
    # ---------------------------
    def _fetch_html(self) -> str | None:
        if not self.valid or not self.url:
            return None
        try:
            resp = requests.get(self.url, timeout=10)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None

    def _clean_html_for_tables(self, html: str) -> str:
        """Remove <sup> citations and elements with style='display:none'."""
        soup = BeautifulSoup(html, "lxml")

        # Remove citations/footnotes
        for sup in soup.find_all("sup"):
            sup.decompose()

        # Remove hidden nodes
        for elem in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "").lower()):
            elem.decompose()

        return str(soup)

    # ---------------------------
    # Public scraping methods
    # ---------------------------
    def scrape(self) -> str | None:
        """Return raw HTML of the Wikipedia page."""
        return self._fetch_html()

    def scrapeTable(self) -> dict[str, pd.DataFrame] | None:
        """
        Scrape tables from the page, clean HTML, and return a dict:
        key = colon-joined header names,
        value = DataFrame where any cell containing multiple $-prefixed values
                is replaced by the maximum $ amount (retaining $ symbol).
        """
        if not self.valid or not self.url:
            return None

        import re
        from io import StringIO

        # Regex for matching $ amounts like $2,212,300,000 or $1.5 billion
        money_re = re.compile(r"\$[\d,]+(?:\.\d+)?")

        def max_dollar_value(cell):
            """Return max $ string if multiple present, else original."""
            import pandas as pd
            if pd.isna(cell):
                return cell
            s = str(cell)
            if "$" not in s:
                return cell
            matches = money_re.findall(s)
            if not matches:
                return cell
            # Convert to numeric for comparison, but keep original strings
            numeric_vals = [float(m.replace("$", "").replace(",", "")) for m in matches]
            max_val = max(numeric_vals)
            # Return the original string corresponding to the max numeric value
            for m in matches:
                if float(m.replace("$", "").replace(",", "")) == max_val:
                    return m
            return cell

        try:
            # Fetch and clean HTML
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            # Remove <sup> tags
            for sup in soup.find_all("sup"):
                sup.decompose()
            # Remove hidden elements
            for elem in soup.find_all(style=lambda st: st and "display:none" in st.replace(" ", "").lower()):
                elem.decompose()

            # Parse tables
            tables = pd.read_html(StringIO(str(soup)))
            if not tables:
                return None

            tables_dict: dict[str, pd.DataFrame] = {}

            for idx, table in enumerate(tables):
                df = table.copy()
                df.columns = [str(c).strip() for c in df.columns]

                # Skip if all headers are numeric
                if all(pd.to_numeric(df.columns, errors="coerce").notna()):
                    continue

                # Clean only cells containing $
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].apply(max_dollar_value)

                # Create unique key from headers
                header_key = ":".join(df.columns)
                key = header_key if header_key not in tables_dict else f"{header_key}:{idx}"
                tables_dict[key] = df

            return tables_dict or None

        except Exception:
            return None

    # ---------------------------
    # Cleaning helpers (generic)
    # ---------------------------
    @staticmethod
    def _to_float_safe(s: str) -> float | None:
        try:
            return float(s)
        except Exception:
            return None

    @classmethod
    def numeric_max(cls, cell: Any) -> float | None:
        """
        Extract all numeric tokens from a messy cell (ranges, parentheses, +, commas)
        and return the MAX as float. No units handling.
        """
        if pd.isna(cell):
            return None
        s = str(cell).replace(",", "")
        nums = []
        for m in cls._NUM_RE.findall(s):
            v = cls._to_float_safe(m)
            if v is not None:
                nums.append(v)
        return float(max(nums)) if nums else None

    @classmethod
    def money_max_usd(cls, cell: Any) -> float | None:
        """
        Extract all amounts (supports 'million'/'billion' units) and return MAX in USD.
        Examples:
          '$50,000,000–100,000,000'   -> 100_000_000
          '$20,000,000+ ($5,200,000)' -> 20_000_000
          '$1.75 billion'             -> 1_750_000_000
        """
        if pd.isna(cell):
            return None
        s = (str(cell)
             .replace("$", "")
             .replace(",", "")
             .replace("\u2013", "-")
             .replace("–", "-")
             .replace("\u2212", "-"))
        vals: list[float] = []
        for num, unit in cls._MONEY_RE.findall(s):
            v = cls._to_float_safe(num)
            if v is None:
                continue
            unit = (unit or "").lower()
            if unit == "million":
                v *= 1_000_000
            elif unit == "billion":
                v *= 1_000_000_000
            vals.append(v)
        return float(max(vals)) if vals else None

    # ---------------------------
    # DataFrame-wide cleaners
    # ---------------------------
    def clean_tables_with_max(
        self,
        tables: dict[str, pd.DataFrame],
        *,
        money_cols: list[str] | None = None,
        generic_cols: list[str] | None = None,
        suffix: str = "_max",
    ) -> dict[str, pd.DataFrame]:
        """
        For each DataFrame in `tables`, add new columns with cleaned MAX values.
          - money_cols: parse via money_max_usd (million/billion aware)
          - generic_cols: parse via numeric_max
        If generic_cols is None, apply numeric_max to all object columns NOT in money_cols.
        """
        cleaned: dict[str, pd.DataFrame] = {}

        for key, df in tables.items():
            out = df.copy()
            # normalize headers once more
            out.columns = [str(c).strip() for c in out.columns]

            # money columns (case-insensitive match)
            if money_cols:
                money_map = {c for c in out.columns for m in money_cols if c.lower() == m.lower()}
            else:
                money_map = set()

            for c in money_map:
                out[c + suffix] = out[c].apply(self.money_max_usd)

            # generic columns default
            if generic_cols is None:
                candidates = [c for c in out.columns if out[c].dtype == "object" and c not in money_map]
            else:
                candidates = [c for c in out.columns for g in generic_cols if c.lower() == g.lower()]

            for c in candidates:
                out[c + suffix] = out[c].apply(self.numeric_max)

            cleaned[key] = out

        return cleaned

    def __repr__(self) -> str:
        return f"Wikipedia(search_query={self.search_query!r}, url={self.url!r}, valid={self.valid}, title={self.title!r})"