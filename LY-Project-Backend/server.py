# server.py
# Requirements:
# pip install fastapi uvicorn requests[socks] beautifulsoup4 stem

import pandas as pd
import uuid
import threading
import time
import random
from queue import Queue, Empty
from typing import Dict
from collections import deque
from urllib.parse import urljoin, quote_plus
import sqlite3
import datetime
import traceback
from nlp_model import classifier as zs_classifier, candidate_labels

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import requests
import re
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller

import joblib
import os
from typing import Optional

# Import NEW feature extractor (blockchain.info based)
try:
    from blockchain_info_extractor import fetch_all_transactions, compute_features
    print("Loaded blockchain.info feature extractor.")
except Exception as ex:
    fetch_all_transactions = None
    compute_features = None
    print("Warning: could not import blockchain_info_extractor:", ex)


# -------------------------
# Tor / crawler configuration
# -------------------------
TOR_SOCKS_PORT = 9050
TOR_CONTROL_PORT = 9051
proxies = {
    'http': f'socks5h://127.0.0.1:{TOR_SOCKS_PORT}',
    'https': f'socks5h://127.0.0.1:{TOR_SOCKS_PORT}'
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/119.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 Safari/537.36",
]

# The Ahmia .onion search host
AHMIA_ONION_HOST = "juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd.onion"
AHMIA_SEARCH_PATH = "/search/"

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session state (for demo/dev)
sessions: Dict[str, Queue] = {}        # session_id -> Queue of log strings
stop_flags: Dict[str, threading.Event] = {}  # session_id -> Event to stop crawl

# In-memory features store: address -> feature dict (thread-safe)
address_features: Dict[str, Dict] = {}
_features_lock = threading.Lock()

# -------------------------
# SQLite storage for extracted addresses
# -------------------------
DB_PATH = "extracted_addresses.db"
_db_lock = threading.Lock()
_db_conn = None

def init_db():
    global _db_conn
    _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    cur = _db_conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS extracted_addresses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        url TEXT,
        depth INTEGER,
        title TEXT,
        address TEXT,
        context_snippet TEXT,
        saved_at TEXT,
        is_illicit INTEGER DEFAULT 0,
        nlp_label TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_address ON extracted_addresses(address)")
    _db_conn.commit()

init_db()


# --- Inference artifacts (scalers, feature order, model) ---
ARTIFACTS = {
    "scalers": None,
    "feature_cols": None,
    "model": None,
    "loaded": False
}

def load_inference_artifacts():
    """Load scalers, feature order, and model if available. Non-fatal if missing."""
    try:
        if ARTIFACTS["loaded"]:
            return
        if os.path.exists("wallet_scalers.pkl"):
            ARTIFACTS["scalers"] = joblib.load("wallet_scalers.pkl")
            print("Loaded wallet_scalers.pkl")
        else:
            print("wallet_scalers.pkl not found")

        if os.path.exists("feature_order.pkl"):
            ARTIFACTS["feature_cols"] = joblib.load("feature_order.pkl")
            print("Loaded feature_order.pkl")
        else:
            print("feature_order.pkl not found")

        if os.path.exists("random_forest_model.pkl"):
            ARTIFACTS["model"] = joblib.load("random_forest_model.pkl")
            print("Loaded random_forest_model.pkl")
        else:
            print("random_forest_model.pkl not found")

    except Exception as ex:
        print("Error loading inference artifacts:", ex, traceback.format_exc())
    ARTIFACTS["loaded"] = True

load_inference_artifacts()


def classify_snippet_zero_shot(text: str) -> str:
    """Classify snippet using zero-shot model. Always returns the best label."""
    try:
        result = zs_classifier(text, candidate_labels)
        return result["labels"][0]
    except Exception as e:
        print("Zero-shot classification failed:", e)
        return "unknown"


def compute_and_store_features(
    session_id: str,
    url: str,
    depth: int,
    title: str,
    address: str,
    context_snippet,
    log_queue: Optional[Queue] = None
):
    """
    Background worker for each discovered Bitcoin address:
        1. NLP classification on the context snippet (always runs)
        2. Fetch full transaction history via blockchain.info API
        3. Compute 55 wallet features
        4. Run Random Forest model for illicit/licit classification
        5. Save everything to SQLite
    """
    try:
        print(f"[DEBUG] Worker started for {address}")
        
        if log_queue:
            log_queue.put("")
            log_queue.put(f"   ── Processing: {address} ──")

        # ── STEP 1: NLP classification (always runs) ──
        try:
            nlp_label = classify_snippet_zero_shot(context_snippet)
            if log_queue:
                log_queue.put(f"   🧠 NLP label: {nlp_label}")
        except Exception as e:
            nlp_label = "unknown"
            if log_queue:
                log_queue.put(f"   ⚠️ Zero-shot NLP failed: {e}")

        # ── STEP 2: Check if feature extractor is available ──
        if fetch_all_transactions is None or compute_features is None:
            if log_queue:
                log_queue.put("   ⚠️ Feature extractor not available")
            with _features_lock:
                address_features[address] = {}
            save_address_to_db(
                session_id, url, depth, title, address,
                context_snippet, is_illicit=0, nlp_label=nlp_label
            )
            return

        # ── STEP 3: Fetch transactions from blockchain.info ──
        api_data = None
        try:
            api_data = fetch_all_transactions(address, limit=50, sleep_sec=1.0)
            if log_queue:
                n = len(api_data.get("txs", []))
                log_queue.put(f"   📦 Fetched {n} transactions")
        except Exception as e:
            if log_queue:
                log_queue.put(f"   ⚠️ Blockchain fetch failed: {e}")
            with _features_lock:
                address_features[address] = {}
            save_address_to_db(
                session_id, url, depth, title, address,
                context_snippet, is_illicit=0, nlp_label=nlp_label
            )
            return

        # ── STEP 4: Compute 55 wallet features ──
        feats = {}
        try:
            feats = compute_features(api_data, address)
            if log_queue:
                log_queue.put(f"   ✅ {len(feats)} features computed")
        except Exception as e:
            if log_queue:
                log_queue.put(f"   ⚠️ Feature computation failed: {e}")
            feats = {}

        with _features_lock:
            address_features[address] = feats

        # ── STEP 5: ML model inference ──
        # Model labels: 0 = licit, 1 = illicit
        # DB convention: is_illicit=0 means licit, is_illicit=1 means illicit
        # These match directly: is_illicit = pred
        is_illicit = 0  # default licit

        try:
            if feats and ARTIFACTS["scalers"] and ARTIFACTS["feature_cols"] and ARTIFACTS["model"]:
                scalers = ARTIFACTS["scalers"]
                feature_cols = ARTIFACTS["feature_cols"]
                model = ARTIFACTS["model"]

                # Build feature row in correct column order
                row = []
                for col in feature_cols:
                    val = feats.get(col, None)
                    if val is None:
                        # Handle space/underscore mismatch
                        alt = col.replace("_", " ") if "_" in col else col.replace(" ", "_")
                        val = feats.get(alt, 0.0)
                    row.append(float(val) if val is not None else 0.0)

                # Scale features
                X_new = pd.DataFrame([row], columns=feature_cols).astype(float)
                for col in feature_cols:
                    try:
                        X_new[col] = scalers[col].transform([[X_new[col].values[0]]])[0][0]
                    except:
                        pass

                # Predict
                pred = model.predict(X_new.values)[0]
                is_illicit = int(pred)  # 0=licit, 1=illicit — matches DB directly

                if log_queue:
                    probs = model.predict_proba(X_new.values)[0] if hasattr(model, "predict_proba") else None
                    prob_str = f" (P(illicit)={probs[1]:.2f})" if probs is not None else ""
                    label_str = "ILLICIT" if is_illicit == 1 else "LICIT"
                    log_queue.put(f"   🔍 ML prediction: {label_str}{prob_str}")

        except Exception as e:
            if log_queue:
                log_queue.put(f"   ⚠️ ML inference failed: {e}")

        # ── STEP 6: Save to database ──
        save_address_to_db(
            session_id, url, depth, title, address,
            context_snippet, is_illicit=is_illicit, nlp_label=nlp_label
        )

        if log_queue:
            log_queue.put(f"   💾 Saved to DB (NLP: {nlp_label} | ML: {'illicit' if is_illicit else 'licit'})")
            log_queue.put("")

    except Exception as e:
        if log_queue:
            log_queue.put(f"   ❌ Unexpected error for {address}: {e}")
        else:
            print("Feature worker error:", e)

        with _features_lock:
            address_features[address] = {}

        try:
            save_address_to_db(
                session_id, url, depth, title, address,
                context_snippet, is_illicit=0, nlp_label="unknown"
            )
        except Exception:
            pass


def save_address_to_db(session_id: str, url: str, depth: int, title: str,
                       address: str, context_snippet: str,
                       is_illicit: int = 0, nlp_label: str = None):
    """Thread-safe insert of a discovered address into SQLite."""
    try:
        with _db_lock:
            cur = _db_conn.cursor()
            cur.execute("""
                INSERT INTO extracted_addresses
                (session_id, url, depth, title, address, context_snippet, saved_at, is_illicit, nlp_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, url, depth, title, address, context_snippet,
                datetime.datetime.utcnow().isoformat(),
                int(is_illicit), nlp_label
            ))
            _db_conn.commit()
    except Exception as e:
        print(f"Error saving address to DB: {e}", traceback.format_exc())


# -------------------------
# Helper functions used by crawler
# -------------------------
def renew_ip(log_queue):
    """Rotate Tor IP via ControlPort (NEWNYM)."""
    try:
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:
            controller.authenticate()
            controller.signal(Signal.NEWNYM)
            log_queue.put("🔄 Tor circuit renewed (new IP).")
            time.sleep(5)
    except Exception as e:
        log_queue.put(f"❌ Could not renew IP: {e}")

def throttle_request(log_queue, min_delay=2, max_delay=6):
    """Add random delay between requests for throttling."""
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)

def fetch_ahmia_search_token(log_queue=None):
    """
    Fetch Ahmia's homepage and extract the hidden CSRF token from the search form.
    Returns the token string, or None if extraction fails.
    """
    try:
        homepage_url = f"http://{AHMIA_ONION_HOST}/"
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        resp = requests.get(homepage_url, proxies=proxies, headers=headers, timeout=20)

        if resp.status_code != 200:
            if log_queue:
                log_queue.put(f"⚠️ Could not fetch Ahmia homepage (status {resp.status_code})")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find hidden input fields in the search form (exclude the query field)
        for inp in soup.find_all("input", attrs={"type": "hidden"}):
            name = inp.get("name", "")
            value = inp.get("value", "")
            if name and value and name != "q":
                return f"{name}={value}"

        if log_queue:
            log_queue.put("⚠️ Could not find search token on Ahmia homepage")
        return None

    except Exception as e:
        if log_queue:
            log_queue.put(f"⚠️ Failed to fetch Ahmia token: {e}")
        return None


def normalize_start_input(raw_input: str, log_queue=None) -> str:
    """
    If raw_input looks like a URL (starts with http/https) return as-is.
    Otherwise treat it as a keyword: fetch Ahmia's CSRF token dynamically,
    then build the search URL.
    """
    s = raw_input.strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        return s

    query = quote_plus(s)
    token = fetch_ahmia_search_token(log_queue)
    if token:
        return f"http://{AHMIA_ONION_HOST}{AHMIA_SEARCH_PATH}?q={query}&{token}"
    else:
        # Fallback: try without token (may redirect to homepage)
        return f"http://{AHMIA_ONION_HOST}{AHMIA_SEARCH_PATH}?q={query}"

def extract_context_snippet(full_text: str, match_span: tuple, window: int = 200) -> str:
    """Grab a snippet of text around the address occurrence (window characters each side)."""
    start, end = match_span
    s = max(0, start - window)
    e = min(len(full_text), end + window)
    snippet = full_text[s:e].strip()
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet

BTC_ADDR_RE = re.compile(r'\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b')

def bfs_crawl_stream(start_url: str, max_depth: int, log_queue: Queue,
                     stop_event: threading.Event, rotate_every=5, session_id=None):
    """
    Breadth-first crawl through Tor.
    For each page: extract Bitcoin addresses, spawn background workers
    for NLP + blockchain feature extraction + ML inference.
    """
    visited = set()
    queue = deque([(start_url, 1)])
    req_count = 0

    log_queue.put(f"═══ CRAWL STARTED ═══")
    log_queue.put(f"   Target: {start_url}")
    log_queue.put(f"   Max depth: {max_depth}")
    log_queue.put("")
    while queue and not stop_event.is_set():
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            resp = requests.get(url, proxies=proxies, headers=headers, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")

            title = soup.title.string.strip() if soup.title else "No title"
            log_queue.put("")  # blank line between pages
            log_queue.put(f"[Depth {depth}] {title}")
            log_queue.put(f"   URL: {url}")

            page_text = soup.get_text(separator=' ', strip=True)

            # Find Bitcoin addresses and start background processing
            for m in BTC_ADDR_RE.finditer(page_text):
                addr = m.group(0)
                snippet = extract_context_snippet(page_text, m.span(), window=200)
                log_queue.put(f"   💰 BTC Address found: {addr}")

                try:
                    th = threading.Thread(
                        target=compute_and_store_features,
                        args=(session_id or "unknown", url, depth, title, addr, snippet, log_queue),
                        daemon=True
                    )
                    th.start()
                except Exception as e:
                    log_queue.put(f"   ❌ Failed to start worker for {addr}: {e}")
                    with _features_lock:
                        address_features[addr] = {}

            # ── Link extraction ──
            # If we're on an Ahmia page, only follow search result redirect links
            # (they contain /search/redirect?...redirect_url=http://actual.onion/...)
            # Skip Ahmia's own navigation (about, privacy, terms, etc.)
            # On all other pages, follow any .onion link.
            is_ahmia_page = AHMIA_ONION_HOST in url

            if is_ahmia_page:
                search_results = []
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    if "/search/redirect" in link and "redirect_url=" in link and link not in visited:
                        queue.append((link, depth + 1))
                        search_results.append(link)
                if search_results:
                    log_queue.put(f"   🔗 Queued {len(search_results)} search results for crawling")
            else:
                # On non-Ahmia pages: follow only .onion links (skip clearnet)
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    if link.startswith("http") and ".onion" in link and link not in visited:
                        queue.append((link, depth + 1))

            req_count += 1
            if req_count % rotate_every == 0:
                renew_ip(log_queue)
            throttle_request(log_queue)

        except Exception as e:
            log_queue.put(f"⚠️ Error fetching {url}: {e}")

    if stop_event.is_set():
        log_queue.put("")
        log_queue.put("═══ CRAWL STOPPED BY USER ═══")
    else:
        log_queue.put("")
        log_queue.put("═══ CRAWL COMPLETE ═══")
        log_queue.put("==DONE==")


# -------------------------
# API endpoints
# -------------------------
@app.post("/start")
def start_crawl(payload: dict):
    """Start a crawl session. Accepts start_url (keyword or URL) and max_depth."""
    raw = payload.get("start_url")
    if not raw:
        raise HTTPException(status_code=400, detail="start_url required")

    session_id = str(uuid.uuid4())
    q = Queue()
    stop_event = threading.Event()

    sessions[session_id] = q
    stop_flags[session_id] = stop_event

    # Build the start URL (fetches Ahmia token if keyword search)
    start_url = normalize_start_input(raw, log_queue=q)

    try:
        max_depth = int(payload.get("max_depth", 2))
    except Exception:
        max_depth = 2

    t = threading.Thread(
        target=bfs_crawl_stream,
        args=(start_url, max_depth, q, stop_event),
        kwargs={'session_id': session_id},
        daemon=True
    )
    t.start()

    return JSONResponse({
        "session_id": session_id,
        "stream_url": f"/stream/{session_id}",
        "start_url": start_url
    })

@app.post("/stop/{session_id}")
def stop_crawl(session_id: str):
    """Signal a running crawl to stop."""
    flag = stop_flags.get(session_id)
    if flag is None:
        raise HTTPException(status_code=404, detail="Session not found")
    flag.set()
    return {"status": "stopping", "session_id": session_id}

def event_generator(session_id: str):
    """Yield server-sent events from the session's log queue."""
    q = sessions.get(session_id)
    if q is None:
        yield "data: ERROR: session not found\n\n"
        return

    while True:
        try:
            line = q.get(timeout=0.5)
            if line == "==DONE==":
                yield f"data: {line}\n\n"
                break
            for chunk in str(line).splitlines():
                yield f"data: {chunk}\n\n"
        except Empty:
            yield ": keep-alive\n\n"
            continue

    sessions.pop(session_id, None)
    stop_flags.pop(session_id, None)

@app.get("/stream/{session_id}")
def stream(session_id: str):
    """SSE endpoint for live crawl logs."""
    headers = {"Content-Type": "text/event-stream"}
    return StreamingResponse(event_generator(session_id), headers=headers)

@app.get("/addresses")
def list_addresses(address: str = Query(None), limit: int = Query(100)):
    """Return saved addresses from the database."""
    try:
        with _db_lock:
            cur = _db_conn.cursor()
            if address:
                cur.execute("""
                    SELECT id, session_id, url, depth, title, address,
                           context_snippet, saved_at, is_illicit, nlp_label
                    FROM extracted_addresses
                    WHERE address = ?
                    ORDER BY id DESC LIMIT ?
                """, (address, limit))
            else:
                cur.execute("""
                    SELECT id, session_id, url, depth, title, address,
                           context_snippet, saved_at, is_illicit, nlp_label
                    FROM extracted_addresses
                    ORDER BY id DESC LIMIT ?
                """, (limit,))

            rows = cur.fetchall()

        return [
            {
                "id": r[0], "session_id": r[1], "url": r[2],
                "depth": r[3], "title": r[4], "address": r[5],
                "context_snippet": r[6], "saved_at": r[7],
                "is_illicit": int(r[8]) if r[8] is not None else 0,
                "nlp_label": r[9] if r[9] is not None else None
            }
            for r in rows
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))