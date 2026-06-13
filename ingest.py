"""Download 10-K filings from SEC EDGAR and reduce them to plain text.

Flow: ticker -> CIK -> latest 10-K accession -> primary document -> text.

SEC requires a descriptive User-Agent with a contact email on every request.
Set it via the SEC_USER_AGENT env var or edit USER_AGENT below.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "rag-eval-harness ohm_patel@brown.edu"
)
HEADERS = {"User-Agent": USER_AGENT}
DATA_DIR = Path(__file__).parent / "data"


def _get(url: str) -> requests.Response:
    """GET with SEC headers and a polite delay (SEC fair-access ~10 req/s)."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)
    return resp


def ticker_to_cik(ticker: str) -> str:
    """Resolve a ticker symbol to a zero-padded 10-digit CIK."""
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    ticker = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == ticker:
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker!r} not found in SEC company list")


def latest_10k(cik: str) -> tuple[str, str]:
    """Return (accession_no_dashes, primary_document) for the newest 10-K."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    recent = _get(url).json()["filings"]["recent"]
    for form, acc, doc in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "10-K":
            return acc.replace("-", ""), doc
    raise ValueError(f"No 10-K filing found for CIK {cik}")


def _html_to_text(html: str) -> str:
    """Strip a filing's HTML to readable text, dropping scripts and tables-noise."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"(\xa0|&#160;)", " ", text)  # non-breaking spaces
    return text.strip()


def fetch_10k_text(ticker: str, *, cache: bool = True) -> str:
    """Download the latest 10-K for a ticker and return its plain text.

    Caches the cleaned text under data/<TICKER>_10k.txt so repeated runs and
    the benchmark grid don't re-hit EDGAR.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache_path = DATA_DIR / f"{ticker.upper()}_10k.txt"
    if cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    cik = ticker_to_cik(ticker)
    accession, doc = latest_10k(cik)
    # CIK in the archive path is unpadded.
    cik_int = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"
    text = _html_to_text(_get(url).text)

    if cache:
        cache_path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    body = fetch_10k_text(symbol)
    print(f"{symbol}: {len(body):,} chars")
    print(body[:500], "...")
