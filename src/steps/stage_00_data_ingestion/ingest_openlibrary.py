from pathlib import Path
import pandas as pd
import time , os , sys , random , csv
from typing import Dict, List, Optional, Set , Iterable
from urllib.parse import urljoin
import urllib.robotparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from src.logger.log import logging
from src.config.configuration import AppConfiguration
from src.exception.exception_handler import AppException 
from src.constant import * 


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
        logging.warning("Could not fetch robots.txt (%s). Proceeding cautiously. Error: %s", robots_url, e)
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
    logging.info("Saved %s rows to %s", len(df), path)


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
        logging.debug("Title/author parse failed for %s: %s", source_url, e)

    # ISBN
    try:
        isbn_dd = soup.find("dd", class_="object", itemprop="isbn")
        if isbn_dd:
            out["ISBN"] = isbn_dd.get_text(strip=True)
    except Exception as e:
        logging.debug("ISBN parse failed for %s: %s", source_url, e)

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
        logging.debug("Year/Publisher parse failed for %s: %s", source_url, e)

    # Description
    try:
        read_more = soup.find("div", class_="read-more__content markdown-content")
        if read_more:
            paragraphs = read_more.find_all("p")
            out["Description"] = " ".join(p.get_text(strip=True) for p in paragraphs)
    except Exception as e:
        logging.debug("Description parse failed for %s: %s", source_url, e)

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
        logging.debug("Image parse failed for %s: %s", source_url, e)

    return out


# -----------------------
# Scraping functions
# -----------------------
def fetch_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Fetch a URL and return BeautifulSoup object; raises on HTTP errors."""
    logging.debug("Fetching URL: %s", url)
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scrape_book(session: requests.Session, url: str, subject: Optional[str] = None) -> Optional[Dict[str, Optional[str]]]:
    """Scrape a single book page and return data dict or None on failure."""
    try:
        if not is_allowed_by_robots(session, url):
            logging.info("Blocked by robots.txt: %s", url)
            return None

        soup = fetch_soup(session, url)
        data = parse_book_page(soup, source_url=url, subject=subject)
        random_delay()
        return data
    except requests.HTTPError as e:
        logging.warning("HTTP error while scraping %s: %s", url, e)
    except Exception as e:
        logging.exception("Unexpected error while scraping %s: %s", url, e)
    return None


def get_work_urls_from_search(session: requests.Session, query: str, page: int = 1) -> Set[str]:
    """Return a set of work URLs for a search query & page."""
    search_url = f"{BASE}/search?q={query}&page={page}"
    logging.info("Searching %s (page=%d)", query, page)
    try:
        soup = fetch_soup(session, search_url)
        urls: Set[str] = set()
        for a in soup.select('a[href^="/works/OL"]'):
            href = a.get("href", "")
            urls.add(urljoin(BASE, href.split("?")[0]))
        random_delay()
        return urls
    except Exception as e:
        logging.exception("Failed to get work URLs for query=%s page=%d: %s", query, page, e)
        return set()


def get_books_subjects(session: requests.Session) -> List[str]:
    """Return list of subject strings from the subjects page."""
    subjects_page = urljoin(BASE, "/subjects")
    logging.info("Fetching subjects from %s", subjects_page)
    try:
        soup = fetch_soup(session, subjects_page)
        body = soup.find("div", id="subjectsPage")
        if not body:
            return []
        return [li.get_text(strip=True) for li in body.find_all("li")]
    except Exception as e:
        logging.exception("Failed to fetch subjects: %s", e)
        return []


# -----------------------
# Data storage (CSV/Parquet) helpers
# -----------------------
def load_existing_rows(path: Path) -> Dict[str, Dict]:
    """Load existing rows indexed by ISBN (or other unique key)."""
    rows = {}
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        for _, r in df.iterrows():
            isbn = r.get("ISBN")
            if isbn:
                rows[isbn] = r.to_dict()
    except Exception:
        logging.exception("Error loading existing CSV; continuing with empty store.")
    return rows


def upsert_row_and_save(row: Dict, path: Path) -> None:
    """Upsert a single scraped row into CSV file (keeps non-empty fields)."""
    if  row.get("ISBN") is None :
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

    with open(path , "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows.values())


# -----------------------
# Orchestration / Main pipeline
# -----------------------
def collect_data(session: requests.Session, path , pages: range = range(1, 3), subjects_sample_size: int = 5):
    """High-level orchestration: fetch subjects -> for each subject sample queries -> scrape results."""
    subjects = get_books_subjects(session)
    if not subjects:
        logging.warning("No subjects found — aborting collection.")
        return

    for page in pages:
        # sample a subset of subjects to avoid making this run huge
        sample_subjects = random.sample(subjects, min(subjects_sample_size, len(subjects)))
        logging.info("Page %d: sampled %d subjects", page, len(sample_subjects))
        for subj in sample_subjects:
            try:
                work_urls = get_work_urls_from_search(session, subj, page)
                logging.info("Found %d works for subject=%s page=%d", len(work_urls), subj, page)
                for work_url in work_urls:
                    data = scrape_book(session, work_url, subject=subj)
                    if data:
                        upsert_row_and_save(data, path)
            except Exception:
                logging.exception("Error collecting data for subject=%s page=%d", subj, page)





class DataIngestion : 
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """
    def __init__(self , app_config = AppConfiguration() , pages: Iterable[int] = range(1, 5), subjects_sample_size: int = 10) -> None :
        """Initialize the data ingestion class."""
        try : 
            self.data_ingestion_config = app_config.get_data_ingestion_config()
            data_dir =  Path(self.data_ingestion_config.Openlibrary_data_dir)
            data_dir.mkdir(parents=True , exist_ok=True)
            session = create_session()
            allowed = is_allowed_by_robots(session, BASE)
            logging.info("Scraper allowed by robots.txt for base=%s ? %s", BASE, allowed)
            collect_data(session, self.data_ingestion_config.Openlibrary_books, pages=pages, subjects_sample_size=subjects_sample_size)
            logging.info("OpenLibrary data collected at %s", self.data_ingestion_config.Openlibrary_books)
            logging.info("Finished scraping run.")
            openlibrary_books = pd.read_csv(self.data_ingestion_config.Openlibrary_books)
            openlibrary_books.rename(columns={
                                     'Subject':'Categories',
                                     'Image-URL-M':'Image',
                                  }, inplace=True)
            openlibrary_books = openlibrary_books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher' , 'Description' , 'Categories', 'Image']]
            openlibrary_books["ISBN"] = openlibrary_books["ISBN"].str.split("\n")
            openlibrary_books["ISBN"] = openlibrary_books["ISBN"].str[0].str.strip().str.strip(',')
            self.df = openlibrary_books
            logging.info("openlibrarybooks are ready for merging")
        except Exception as e : 
            raise AppException(e , sys) from e 
    def get_data(self) -> pd.DataFrame : 
        return self.df


