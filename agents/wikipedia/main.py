import wikipedia
import requests


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
                # Try to get the title from the Wikipedia library
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

    def scrape(self) -> str | None:
        """Scrape the content of the Wikipedia page."""
        if not self.valid or not self.url:
            return None

        try:
            data = requests.get(self.url, timeout=10)
            return data.text
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"Wikipedia(search_query={self.search_query!r}, url={self.url!r}, valid={self.valid}, title={self.title!r})"
