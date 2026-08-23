"""SEC EDGAR client for insider (Form 4) and institutional (13F) activity.

EDGAR is free and needs no API key — only a declared User-Agent and a
courteous request rate. That makes it the one genuinely "smart money" data
source available to this app, alongside the price/volume proxies in
``common.smart_money``.

Two caveats the UI must keep visible:

* 13F holdings are filed up to 45 days after quarter end, so they confirm a
  theme rather than front-run it.
* Only Form 4 transaction code ``P`` is an open-market purchase. Codes ``A``
  (grant), ``M`` (option exercise) and ``F`` (tax withholding) are
  compensation mechanics, not conviction, and are filtered out.

Parsing helpers are pure functions so they can be unit tested against XML
fixtures; only the network wrappers touch Streamlit's cache.
"""

import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# SEC's fair-access policy requires a User-Agent carrying a contact address,
# and rejects strings containing a URL instead of an email. Deployers should
# point SEC_CONTACT_EMAIL at a mailbox they actually read; the placeholder
# below keeps the app working out of the box without publishing anyone's
# personal address in source control.
DEFAULT_CONTACT_EMAIL = "portfolio-pulse@example.com"


def _user_agent():
    """Identify this client to SEC, honouring a deployer-supplied contact."""
    contact = os.environ.get("SEC_CONTACT_EMAIL", "").strip() or DEFAULT_CONTACT_EMAIL
    return f"PortfolioPulse/1.0 ({contact})"


# SEC asks for no more than 10 requests/second; stay comfortably under.
MAX_REQUESTS_PER_SECOND = 8
REQUEST_TIMEOUT = 20

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# 13F "value" switched from thousands of dollars to whole dollars for periods
# from 2023 onward (SEC release 34-95148).
VALUE_IN_DOLLARS_FROM = date(2023, 1, 1)

INSIDER_BUY_CODE = "P"

# Verified against data.sec.gov/submissions — each entity name confirmed and
# each is an active 13F-HR filer.
CURATED_MANAGERS = {
    1067983: "Berkshire Hathaway",
    1350694: "Bridgewater Associates",
    1037389: "Renaissance Technologies",
    1423053: "Citadel Advisors",
    1167483: "Tiger Global Management",
    1135730: "Coatue Management",
    1536411: "Duquesne Family Office",
    1656456: "Appaloosa",
    1061165: "Lone Pine Capital",
    1103804: "Viking Global Investors",
    1040273: "Third Point",
    1603466: "Point72 Asset Management",
    1273087: "Millennium Management",
    1697748: "ARK Investment Management",
    1029160: "Soros Fund Management",
}

# Corporate-form noise stripped before matching a 13F issuer name to a ticker.
_NAME_NOISE = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COS",
    "LTD", "LIMITED", "PLC", "LP", "LLC", "LLP", "NV", "SA", "AG", "AB",
    "HOLDINGS", "HOLDING", "HLDGS", "HLDG", "GROUP", "GRP", "THE", "TR",
    "TRUST", "COM", "COMMON", "STK", "SHS", "ADR", "ADS", "SPONSORED",
    "SPON", "NEW", "DEL", "CLASS", "CL", "SER", "ORD", "PAR", "REIT",
}
_CLASS_SUFFIX = re.compile(r"\b(CL|CLASS|SER|SERIES)\s+[A-Z]\b")

_rate_lock = threading.Lock()
_last_request = 0.0


def _throttle():
    """Space requests out so the client stays inside SEC's rate limit."""
    global _last_request
    with _rate_lock:
        gap = time.monotonic() - _last_request
        minimum = 1.0 / MAX_REQUESTS_PER_SECOND
        if gap < minimum:
            time.sleep(minimum - gap)
        _last_request = time.monotonic()


@st.cache_resource
def _session():
    """Requests session carrying the User-Agent SEC requires.

    No explicit Accept-Encoding: requests advertises only codings it can
    decode, and hardcoding "br" breaks without the optional brotli package.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _user_agent()})
    return session


def _get(url, max_retries=3, backoff=1.5):
    """Throttled GET returning the response body, or None on failure.

    EDGAR throttles by IP and answers a burst with 403 rather than 429, so
    both are retried with backoff. Any other status is a real miss (a filing
    directory that does not exist) and returns immediately.
    """
    delay = backoff
    for attempt in range(max_retries):
        try:
            _throttle()
            response = _session().get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.content
            if response.status_code not in (403, 429, 503):
                return None
        except requests.RequestException:
            pass
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
    return None


# --- Pure parsing helpers ------------------------------------------------

def _local(tag):
    """Element tag without its XML namespace."""
    return tag.split("}")[-1]


def _text(element, name):
    """Text of the first descendant with this (namespace-agnostic) tag name.

    Ownership forms wrap most transaction fields in a <value> child —
    <transactionShares><value>100</value></transactionShares> — so unwrap
    that level when it is present, otherwise read the element's own text.
    """
    for child in element.iter():
        if _local(child.tag) != name:
            continue
        for grandchild in child:
            if _local(grandchild.tag) == "value":
                return (grandchild.text or "").strip()
        return (child.text or "").strip()
    return None


def normalize_issuer_name(name):
    """Reduce a company name to a comparable key.

    13F info tables spell issuers differently from EDGAR's company index
    ("NVIDIA CORPORATION" vs "NVIDIA Corp"), so both sides are stripped of
    punctuation, share-class markers and corporate-form suffixes.
    """
    if not name:
        return ""
    text = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())
    text = _CLASS_SUFFIX.sub(" ", text)
    words = [w for w in text.split() if w and w not in _NAME_NOISE]
    return " ".join(words)


def parse_form4_xml(data):
    """Extract non-derivative transactions from one Form 4 filing.

    Returns one record per transaction with its code, size and price. The
    caller decides which codes matter; nothing is filtered here.
    """
    if not data:
        return []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []

    symbol = _text(root, "issuerTradingSymbol")
    owner = _text(root, "rptOwnerName")
    is_director = _text(root, "isDirector") in ("1", "true")
    is_officer = _text(root, "isOfficer") in ("1", "true")
    title = _text(root, "officerTitle")

    records = []
    for node in root.iter():
        if _local(node.tag) != "nonDerivativeTransaction":
            continue
        code = _text(node, "transactionCode")
        try:
            shares = float(_text(node, "transactionShares") or "nan")
            price = float(_text(node, "transactionPricePerShare") or "nan")
        except ValueError:
            continue
        records.append({
            "ticker": symbol,
            "owner": owner,
            "role": title or ("Director" if is_director else "Officer" if is_officer else ""),
            "code": code,
            "acquired": _text(node, "transactionAcquiredDisposedCode"),
            "shares": shares,
            "price": price,
            "value": shares * price if pd.notna(shares) and pd.notna(price) else float("nan"),
            "transaction_date": _text(node, "transactionDate"),
        })
    return records


def parse_infotable_xml(data):
    """Parse a 13F information table into a positions DataFrame."""
    columns = ["issuer", "cusip", "value", "shares"]
    if not data:
        return pd.DataFrame(columns=columns)
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return pd.DataFrame(columns=columns)

    rows = []
    for node in root.iter():
        if _local(node.tag) != "infoTable":
            continue
        try:
            value = float(_text(node, "value") or "nan")
        except ValueError:
            value = float("nan")
        try:
            shares = float(_text(node, "sshPrnamt") or "nan")
        except ValueError:
            shares = float("nan")
        rows.append({
            "issuer": _text(node, "nameOfIssuer"),
            "cusip": (_text(node, "cusip") or "").upper(),
            "value": value,
            "shares": shares,
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    # One manager can report the same issuer across several accounts.
    return (
        pd.DataFrame(rows)
        .groupby(["issuer", "cusip"], as_index=False)[["value", "shares"]]
        .sum()
    )


def normalize_position_values(positions, report_date):
    """Put 13F values in whole dollars regardless of filing vintage."""
    if positions.empty:
        return positions
    scaled = positions.copy()
    if report_date and report_date < VALUE_IN_DOLLARS_FROM:
        scaled["value"] = scaled["value"] * 1000
    return scaled


def build_issuer_ticker_map(titles_by_ticker):
    """Normalized company name -> ticker, for the universe we care about."""
    mapping = {}
    for ticker, title in titles_by_ticker.items():
        key = normalize_issuer_name(title)
        if key:
            mapping.setdefault(key, ticker)
    return mapping


def map_positions_to_tickers(positions, issuer_map):
    """Attach a ticker to each 13F position where the issuer name resolves.

    Unmatched rows keep a null ticker rather than being dropped, so the page
    can show how much value went unmapped instead of quietly losing it.
    """
    if positions.empty:
        return positions.assign(ticker=pd.Series(dtype=object))
    resolved = positions.copy()
    resolved["ticker"] = [
        issuer_map.get(normalize_issuer_name(name)) for name in resolved["issuer"]
    ]
    return resolved


def diff_positions(current, prior):
    """Quarter-over-quarter change per position.

    Classifies each holding as a brand-new stake, an add, a trim or an exit —
    new stakes and adds being the ones that signal fresh conviction.
    """
    columns = ["issuer", "cusip", "ticker", "value", "prior_value", "value_change", "action"]
    if current is None or current.empty:
        return pd.DataFrame(columns=columns)

    prior_by_cusip = (
        prior.set_index("cusip")["value"] if prior is not None and not prior.empty
        else pd.Series(dtype=float)
    )
    merged = current.copy()
    merged["prior_value"] = merged["cusip"].map(prior_by_cusip).fillna(0.0)
    merged["value_change"] = merged["value"] - merged["prior_value"]

    def label(row):
        if row["prior_value"] == 0 and row["value"] > 0:
            return "New"
        if row["value_change"] > 0:
            return "Added"
        if row["value_change"] < 0:
            return "Trimmed"
        return "Held"

    merged["action"] = merged.apply(label, axis=1)
    return merged.reindex(columns=columns)


def summarize_insider_buys(transactions, since=None):
    """Open-market insider purchases only, newest first.

    Grants, option exercises and tax-withholding sales are dropped: they are
    scheduled compensation events and say nothing about conviction.
    """
    columns = ["ticker", "owner", "role", "transaction_date", "shares", "price", "value"]
    if not transactions:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(transactions)
    if df.empty or "code" not in df.columns:
        return pd.DataFrame(columns=columns)

    buys = df[(df["code"] == INSIDER_BUY_CODE) & (df["acquired"] == "A")].copy()
    if buys.empty:
        return pd.DataFrame(columns=columns)

    buys["transaction_date"] = pd.to_datetime(buys["transaction_date"], errors="coerce")
    buys = buys.dropna(subset=["transaction_date"])
    if since is not None:
        buys = buys[buys["transaction_date"] >= pd.Timestamp(since)]
    if buys.empty:
        return pd.DataFrame(columns=columns)
    return (
        buys.reindex(columns=columns)
        .sort_values("transaction_date", ascending=False)
        .reset_index(drop=True)
    )


# --- Network wrappers ----------------------------------------------------

@st.cache_data(ttl=86400)
def get_ticker_cik_map():
    """Ticker -> {cik, title} for every EDGAR-registered company."""
    payload = _get(COMPANY_TICKERS_URL)
    if payload is None:
        return {}
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return {
        str(entry["ticker"]).upper(): {
            "cik": int(entry["cik_str"]), "title": str(entry["title"]),
        }
        for entry in raw.values()
        if isinstance(entry, dict) and entry.get("ticker")
    }


@st.cache_data(ttl=21600)
def get_recent_filings(cik, forms):
    """Recent filings of the given form types for one CIK."""
    columns = ["form", "accession", "filing_date", "report_date", "document"]
    payload = _get(SUBMISSIONS_URL.format(cik=int(cik)))
    if payload is None:
        return pd.DataFrame(columns=columns)
    try:
        recent = json.loads(payload.decode("utf-8")).get("filings", {}).get("recent", {})
    except (ValueError, UnicodeDecodeError, AttributeError):
        return pd.DataFrame(columns=columns)
    if not recent or "form" not in recent:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame({
        "form": recent.get("form", []),
        "accession": recent.get("accessionNumber", []),
        "filing_date": recent.get("filingDate", []),
        "report_date": recent.get("reportDate", []),
        "document": recent.get("primaryDocument", []),
    })
    return df[df["form"].isin(list(forms))].reset_index(drop=True)


@st.cache_data(ttl=21600)
def get_insider_buys(tickers, days=90, max_filings_per_ticker=40):
    """Open-market insider purchases across a small set of tickers.

    Scoped intentionally: one theme's constituents at a time. Scanning the
    whole universe would mean thousands of Form 4 fetches.
    """
    cik_map = get_ticker_cik_map()
    if not cik_map:
        return pd.DataFrame()

    since = date.today() - timedelta(days=days)
    transactions = []
    for ticker in tickers:
        entry = cik_map.get(str(ticker).upper())
        if entry is None:
            continue
        cik = entry["cik"]
        filings = get_recent_filings(cik, ["4"])
        if filings.empty:
            continue
        filings = filings[filings["filing_date"] >= since.isoformat()]
        for _, filing in filings.head(max_filings_per_ticker).iterrows():
            document = str(filing["document"])
            # primaryDocument often points at the XSL-rendered view; the raw
            # XML sits alongside it without the stylesheet directory prefix.
            document = document.rsplit("/", 1)[-1]
            url = ARCHIVES_URL.format(
                cik=cik, accession=str(filing["accession"]).replace("-", ""),
            ) + "/" + document
            transactions.extend(parse_form4_xml(_get(url)))

    return summarize_insider_buys(transactions, since=since)


@st.cache_data(ttl=86400)
def get_manager_positions(cik):
    """Latest and prior 13F holdings for one manager, values in dollars.

    Returns (current, prior, report_date, prior_report_date).
    """
    empty = pd.DataFrame(columns=["issuer", "cusip", "value", "shares"])
    filings = get_recent_filings(cik, ["13F-HR"])
    if filings.empty:
        return empty, empty, None, None

    filings = filings.sort_values("report_date", ascending=False)
    snapshots = []
    for _, filing in filings.head(2).iterrows():
        accession = str(filing["accession"]).replace("-", "")
        base = ARCHIVES_URL.format(cik=int(cik), accession=accession)
        listing = _get(base + "/")
        table_name = "infotable.xml"
        if listing is not None:
            names = re.findall(r'href="[^"]*/([^"/]+\.xml)"', listing.decode("utf-8", "ignore"))
            candidates = [n for n in names if n.lower() != "primary_doc.xml"]
            if candidates:
                table_name = candidates[0]
        positions = parse_infotable_xml(_get(f"{base}/{table_name}"))
        report_date = pd.to_datetime(filing["report_date"], errors="coerce")
        snapshots.append((
            normalize_position_values(
                positions, report_date.date() if pd.notna(report_date) else None,
            ),
            filing["report_date"],
        ))

    current, current_date = snapshots[0] if snapshots else (empty, None)
    prior, prior_date = snapshots[1] if len(snapshots) > 1 else (empty, None)
    return current, prior, current_date, prior_date
