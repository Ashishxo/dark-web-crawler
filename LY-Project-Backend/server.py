# server.py
# Requirements:
# pip install fastapi uvicorn requests[socks] beautifulsoup4 stem
# server.py
# Requirements:
# pip install fastapi uvicorn requests[socks] beautifulsoup4 stem

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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import requests
import re
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller

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

# -------------------------
# SQLite storage for extracted addresses
# -------------------------
DB_PATH = "extracted_addresses.db"
_db_lock = threading.Lock()
_db_conn = None

def init_db():
    global _db_conn
    # enable multithreaded access; we'll use a Lock to serialize writes
    _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    cur = _db_conn.cursor()
    # create simple table to store extracted addresses and context
    cur.execute("""
    CREATE TABLE IF NOT EXISTS extracted_addresses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        url TEXT,
        depth INTEGER,
        title TEXT,
        address TEXT,
        context_snippet TEXT,
        saved_at TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_address ON extracted_addresses(address)")
    _db_conn.commit()

# initialize DB at import time
init_db()

def save_address_to_db(session_id: str, url: str, depth: int, title: str, address: str, context_snippet: str):
    """Thread-safe insert of a discovered address and its context into SQLite."""
    try:
        with _db_lock:
            cur = _db_conn.cursor()
            cur.execute("""
                INSERT INTO extracted_addresses (session_id, url, depth, title, address, context_snippet, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                url,
                depth,
                title,
                address,
                context_snippet,
                datetime.datetime.utcnow().isoformat()
            ))
            _db_conn.commit()
    except Exception as e:
        # do not raise — log to console (or to the crawl log) in production
        print(f"Error saving address to DB: {e}")

# -------------------------
# Helper functions used by crawler
# -------------------------
def renew_ip(log_queue):
    """Rotate Tor IP via ControlPort (NEWNYM)."""
    try:
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:
            controller.authenticate()  # adapt if you use password/cookie auth
            controller.signal(Signal.NEWNYM)
            log_queue.put("🔄 Tor circuit renewed (new IP).")
            time.sleep(5)  # allow time for circuit to change
    except Exception as e:
        log_queue.put(f"❌ Could not renew IP: {e}")

def throttle_request(log_queue, min_delay=3, max_delay=7):
    """Add random delay between requests for throttling."""
    delay = random.uniform(min_delay, max_delay)
    log_queue.put(f"⏳ Sleeping {delay:.2f}s before next request...")
    time.sleep(delay)

def normalize_start_input(raw_input: str) -> str:
    """
    If raw_input looks like a URL (starts with http/https) return as-is.
    Otherwise treat it as a keyword and build an Ahmia .onion search URL.
    """
    s = raw_input.strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        return s
    query = quote_plus(s)
    return f"http://{AHMIA_ONION_HOST}{AHMIA_SEARCH_PATH}?q={query}"

def extract_context_snippet(full_text: str, match_span: tuple, window: int = 200) -> str:
    """Grab a snippet of text around the address occurrence (window characters each side)."""
    start, end = match_span
    s = max(0, start - window)
    e = min(len(full_text), end + window)
    snippet = full_text[s:e].strip()
    # normalize whitespace
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet

BTC_ADDR_RE = re.compile(r'\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b')

def bfs_crawl_stream(start_url: str, max_depth: int, log_queue: Queue, stop_event: threading.Event, rotate_every=5, session_id=None):
    """Breadth-first crawl that pushes log lines into log_queue. Checks stop_event to exit early."""
    visited = set()
    queue = deque([(start_url, 1)])
    req_count = 0

    log_queue.put(f"STARTING crawl: {start_url} (max_depth={max_depth})")
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
            log_queue.put(f"[{depth}] {url} — Title: {title}")

            page_text = soup.get_text(separator=' ', strip=True)

            # Find BTC addresses and save each with context
            for m in BTC_ADDR_RE.finditer(page_text):
                addr = m.group(0)
                snippet = extract_context_snippet(page_text, m.span(), window=200)
                log_queue.put(f"   💰 BTC Address found: {addr}")
                # save to DB (session_id optional)
                try:
                    save_address_to_db(session_id or "unknown", url, depth, title, addr, snippet)
                    log_queue.put(f"   ✅ Saved address to DB: {addr}")
                except Exception as e:
                    log_queue.put(f"   ❌ Failed to save {addr}: {e}")

            # Add links to queue
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if link.startswith("http") and link not in visited:
                    queue.append((link, depth + 1))

            req_count += 1
            if req_count % rotate_every == 0:
                renew_ip(log_queue)
            throttle_request(log_queue)

        except Exception as e:
            log_queue.put(f"⚠️ Error fetching {url}: {e}")

    if stop_event.is_set():
        log_queue.put("🛑 Crawl stopped by user.")
    else:
        log_queue.put("==DONE==")

    return

# -------------------------
# API endpoints
# -------------------------
@app.post("/start")
def start_crawl(payload: dict):
    """
    Start a crawl session.
    payload expected fields:
      - start_url: either an absolute URL or a keyword (string)
      - max_depth: optional int (defaults to 2)
    """
    raw = payload.get("start_url")
    if not raw:
        raise HTTPException(status_code=400, detail="start_url required")

    start_url = normalize_start_input(raw)

    try:
        max_depth = int(payload.get("max_depth", 2))
    except Exception:
        max_depth = 2

    session_id = str(uuid.uuid4())
    q = Queue()
    stop_event = threading.Event()

    sessions[session_id] = q
    stop_flags[session_id] = stop_event

    # start crawler in background thread (pass session_id so saved rows reference it)
    t = threading.Thread(target=bfs_crawl_stream, args=(start_url, max_depth, q, stop_event), kwargs={'session_id': session_id}, daemon=True)
    t.start()

    return JSONResponse({"session_id": session_id, "stream_url": f"/stream/{session_id}", "start_url": start_url})

@app.post("/stop/{session_id}")
def stop_crawl(session_id: str):
    """Signal a running crawl to stop."""
    flag = stop_flags.get(session_id)
    if flag is None:
        raise HTTPException(status_code=404, detail="Session not found")
    flag.set()
    return {"status": "stopping", "session_id": session_id}

def event_generator(session_id: str):
    """Yield server-sent events reading from the session queue."""
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
            # send each line as an SSE data chunk
            for chunk in str(line).splitlines():
                yield f"data: {chunk}\n\n"
        except Empty:
            # comment ping to keep connection alive
            yield ": keep-alive\n\n"
            continue

    # cleanup
    sessions.pop(session_id, None)
    stop_flags.pop(session_id, None)
    return

@app.get("/stream/{session_id}")
def stream(session_id: str):
    """SSE endpoint for client to connect to for live logs."""
    headers = {"Content-Type": "text/event-stream"}
    return StreamingResponse(event_generator(session_id), headers=headers)

# -------------------------
# Simple API to retrieve saved addresses
# -------------------------
@app.get("/addresses")
def list_addresses(address: str = Query(None), limit: int = Query(100)):
    """
    Return saved addresses.
    Optional query params:
      - address: filter by exact address
      - limit: maximum rows
    """
    try:
        with _db_lock:
            cur = _db_conn.cursor()
            if address:
                cur.execute("SELECT id, session_id, url, depth, title, address, context_snippet, saved_at FROM extracted_addresses WHERE address = ? ORDER BY id DESC LIMIT ?", (address, limit))
            else:
                cur.execute("SELECT id, session_id, url, depth, title, address, context_snippet, saved_at FROM extracted_addresses ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        # map to objects
        result = []
        for r in rows:
            result.append({
                "id": r[0],
                "session_id": r[1],
                "url": r[2],
                "depth": r[3],
                "title": r[4],
                "address": r[5],
                "context_snippet": r[6],
                "saved_at": r[7]
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))