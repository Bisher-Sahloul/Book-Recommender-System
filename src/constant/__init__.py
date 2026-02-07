import os 
from urllib.parse import urljoin
import urllib.robotparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT_DIR = os.getcwd()
CONFIG_FOLDER_NAME = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR , CONFIG_FOLDER_NAME , CONFIG_FILE_NAME)


BASE = "https://openlibrary.org"
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




TOP_K = 10 