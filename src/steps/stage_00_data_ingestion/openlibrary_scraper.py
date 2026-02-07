# stage_00_data_ingestion/openlibrary_scraper.py
import time
import random
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin
import urllib.robotparser
from src.logger.log import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd

# -----------------------
# Configuration
# -----------------------
BASE = "https://openlibrary.org"
DATA_DIR = Path("data/raw/openlibrary")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "openlibrary_books.csv"
FIELDNAMES = [
    "ISBN",
    "Book-Title",
    "Book-Author",
    "Year-Of-Publication",
    "Publisher",
    "Subject",
    "Description",
    "Image-URL-S",
    "Image-URL-M",
    "Image-URL-L",
    "Source-URL",
]
USER_AGENT = "OpenLibraryResearchBot/1.0 (+https://github.com/Bisher-Sahloul)"
MIN_DELAY = 1.0  # seconds
MAX_DELAY = 3.0  # seconds
REQUEST_TIMEOUT = 15  # seconds

# Retry strategy for requests
RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
)

# -----------------------
# Logging setup
# -----------------------
LOG_FILE = DATA_DIR / "scraper.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("openlibrary_scraper")

# -----------------------
# HTTP session
# -----------------------
def create_session() -> requests.Session:
    """Create a requests session with retry and a user-agent header."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


# -----------------------
# Robots.txt check (politeness)
# -----------------------
def is_allowed_by_robots(session: requests.Session, target_url: str) -> bool:
    """
    Check robots.txt for permission to scrape the given URL.
    Returns True if allowed or if robots.txt cannot be fetched.
    """
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(BASE, "/robots.txt")
    try:
        resp = session.get(robots_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        rp.parse(resp.text.splitlines())
        return rp.can_fetch(USER_AGENT, target_url)
    except Exception as e:
        # If robots.txt cannot be retrieved, be conservative and allow,
        # but log a warning. You might want to refuse in stricter settings.
        logger.warning("Could not fetch robots.txt (%s). Proceeding cautiously. Error: %s", robots_url, e)
        return True


# -----------------------
# Utilities
# -----------------------
def random_delay():
    """Sleep a random interval between MIN_DELAY and MAX_DELAY."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def atomic_write_df(df: pd.DataFrame, path: Path, fmt: str = "csv") -> None:
    """Write a DataFrame atomically to disk (csv or parquet)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    if fmt == "csv":
        df.to_csv(tmp, index=False, encoding="utf-8")
    elif fmt == "parquet":
        df.to_parquet(tmp, index=False)
    else:
        raise ValueError("Unsupported format: %s" % fmt)
    tmp.replace(path)
    logger.info("Saved %s rows to %s", len(df), path)


# -----------------------
# Parsing logic
# -----------------------
def parse_book_page(soup: BeautifulSoup, source_url: str, subject: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Extract fields from a BeautifulSoup-parsed book page.
    Returns a dict with FIELDNAMES + Source-URL.
    """
    # Defaults
    out = {k: None for k in FIELDNAMES}
    out["Source-URL"] = source_url
    out["Subject"] = subject

    # Example selectors — update if OL structure changes
    try:
        body = soup.find("div", class_="work-title-and-author mobile")
        if body:
            title_tag = body.select_one("span > h1")
            if title_tag:
                out["Book-Title"] = title_tag.get_text(strip=True)
            author_tag = body.select_one("h2.edition-byline a")
            if author_tag:
                out["Book-Author"] = author_tag.get_text(strip=True)
    except Exception as e:
        logger.debug("Title/author parse failed for %s: %s", source_url, e)

    # ISBN
    try:
        isbn_dd = soup.find("dd", class_="object", itemprop="isbn")
        if isbn_dd:
            out["ISBN"] = isbn_dd.get_text(strip=True)
    except Exception as e:
        logger.debug("ISBN parse failed for %s: %s", source_url, e)

    # Year and Publisher
    try:
        edition_about = soup.find("div", class_="editionAbout")
        if edition_about:
            span_date = edition_about.find("span", itemprop="datePublished")
            if span_date:
                out["Year-Of-Publication"] = span_date.get_text(strip=True)
            publisher_a = edition_about.find("a", itemprop="publisher")
            if publisher_a:
                out["Publisher"] = publisher_a.get_text(strip=True)
    except Exception as e:
        logger.debug("Year/Publisher parse failed for %s: %s", source_url, e)

    # Description
    try:
        read_more = soup.find("div", class_="read-more__content markdown-content")
        if read_more:
            paragraphs = read_more.find_all("p")
            out["Description"] = " ".join(p.get_text(strip=True) for p in paragraphs)
    except Exception as e:
        logger.debug("Description parse failed for %s: %s", source_url, e)

    # Image URLs
    try:
        img = soup.find("img", itemprop="image")
        if img and img.get("src"):
            image_url = img["src"]
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            out["Image-URL-S"] = image_url[:-5] + "S"
            out["Image-URL-M"] = image_url[:-5] + "M"
            out["Image-URL-L"] = image_url[:-5] + "L"
    except Exception as e:
        logger.debug("Image parse failed for %s: %s", source_url, e)

    return out


# -----------------------
# Scraping functions
# -----------------------
def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Fetch a URL and return BeautifulSoup object; raises on HTTP errors."""
    logger.debug("Fetching URL: %s", url)
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scrape_book(session: requests.Session, url: str, subject: Optional[str] = None) -> Optional[Dict[str, Optional[str]]]:
    """Scrape a single book page and return data dict or None on failure."""
    try:
        if not is_allowed_by_robots(session, url):
            logger.info("Blocked by robots.txt: %s", url)
            return None

        soup = fetch_soup(session, url)
        data = parse_book_page(soup, source_url=url, subject=subject)
        random_delay()
        return data
    except requests.HTTPError as e:
        logger.warning("HTTP error while scraping %s: %s", url, e)
    except Exception as e:
        logger.exception("Unexpected error while scraping %s: %s", url, e)
    return None


def get_work_urls_from_search(session: requests.Session, query: str, page: int = 1) -> Set[str]:
    """Return a set of work URLs for a search query & page."""
    search_url = f"{BASE}/search?q={query}&page={page}"
    logger.info("Searching %s (page=%d)", query, page)
    try:
        soup = fetch_soup(session, search_url)
        urls: Set[str] = set()
        for a in soup.select('a[href^="/works/OL"]'):
            href = a.get("href", "")
            urls.add(urljoin(BASE, href.split("?")[0]))
        random_delay()
        return urls
    except Exception as e:
        logger.exception("Failed to get work URLs for query=%s page=%d: %s", query, page, e)
        return set()


def get_books_subjects(session: requests.Session) -> List[str]:
    """Return list of subject strings from the subjects page."""
    subjects_page = urljoin(BASE, "/subjects")
    logger.info("Fetching subjects from %s", subjects_page)
    try:
        soup = fetch_soup(session, subjects_page)
        body = soup.find("div", id="subjectsPage")
        if not body:
            return []
        return [li.get_text(strip=True) for li in body.find_all("li")]
    except Exception as e:
        logger.exception("Failed to fetch subjects: %s", e)
        return []


# -----------------------
# Data storage (CSV/Parquet) helpers
# -----------------------
def load_existing_rows(path: Path) -> Dict[str, Dict]:
    """Load existing rows indexed by ISBN (or other unique key)."""
    rows = {}
    if not path.exists():
        return rows
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        for _, r in df.iterrows():
            isbn = r.get("ISBN")
            if isbn:
                rows[isbn] = r.to_dict()
    except Exception:
        logger.exception("Error loading existing CSV; continuing with empty store.")
    return rows


def upsert_row_and_save(row: Dict, path: Path = CSV_FILE) -> None:
    """Upsert a single scraped row into CSV file (keeps non-empty fields)."""
    if not row or not row.get("ISBN"):
        return
    rows = load_existing_rows(path)
    isbn = row["ISBN"]
    if isbn in rows:
        # Update empty fields
        for k, v in row.items():
            if v and not rows[isbn].get(k):
                rows[isbn][k] = v
    else:
        rows[isbn] = row
    df = pd.DataFrame(list(rows.values()))
    atomic_write_df(df, path, fmt="csv")


# -----------------------
# Orchestration / Main pipeline
# -----------------------
def collect_data(session: requests.Session, pages: range = range(1, 3), subjects_sample_size: int = 5):
    """High-level orchestration: fetch subjects -> for each subject sample queries -> scrape results."""
    subjects = get_books_subjects(session)
    if not subjects:
        logger.warning("No subjects found — aborting collection.")
        return

    for page in pages:
        # sample a subset of subjects to avoid making this run huge
        sample_subjects = random.sample(subjects, min(subjects_sample_size, len(subjects)))
        logger.info("Page %d: sampled %d subjects", page, len(sample_subjects))
        for subj in sample_subjects:
            try:
                work_urls = get_work_urls_from_search(session, subj, page)
                logger.info("Found %d works for subject=%s page=%d", len(work_urls), subj, page)
                for work_url in work_urls:
                    data = scrape_book(session, work_url, subject=subj)
                    if data:
                        upsert_row_and_save(data, CSV_FILE)
            except Exception:
                logger.exception("Error collecting data for subject=%s page=%d", subj, page)


def main():
    session = create_session()
    # Example: check robots first for base
    allowed = is_allowed_by_robots(session, BASE)
    logger.info("Scraper allowed by robots.txt for base=%s ? %s", BASE, allowed)
    # limit pages for initial runs; change as required
    collect_data(session, pages=range(1, 5), subjects_sample_size=10)
    logger.info("Finished scraping run.")


if __name__ == "__main__":
    main()
