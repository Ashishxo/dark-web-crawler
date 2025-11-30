# server.py
# Requirements:
# pip install fastapi uvicorn requests[socks] beautifulsoup4 stem
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

import joblib   # optional for later, harmless now
import os
from typing import Optional
# import feature-extractor functions (will fall back gracefully if module missing)
try:
    from feature_extractor import fetch_blockcypher_full, compute_wallet_features_blockcypher
except Exception as ex:
    fetch_blockcypher_full = None
    compute_wallet_features_blockcypher = None
    print("Warning: could not import blockcypher_wallet_extractor:", ex)

BLOCKCYPHER_TOKEN: Optional[str] = os.environ.get("BLOCKCYPHER_TOKEN", "51eaeb12a21b4a4f85082d5b7c86ec44")


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
    # enable multithreaded access; we'll use a Lock to serialize writes
    _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    cur = _db_conn.cursor()
    # create simple table to store extracted addresses and context, include is_illicit default 0
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
        is_illicit INTEGER DEFAULT 0
    )
    """)
    # create index on address if missing
    cur.execute("CREATE INDEX IF NOT EXISTS idx_address ON extracted_addresses(address)")
    _db_conn.commit()

# initialize DB at import time
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

# call once at import/start
load_inference_artifacts()


def classify_snippet_zero_shot(text: str) -> str:
    """
    Classify snippet using zero-shot model.
    Always returns the best label.
    """
    try:
        result = zs_classifier(text, candidate_labels)
        return result["labels"][0]   # highest score
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
    Background worker:
        - Fetch transactions via BlockCypher
        - Compute wallet features
        - Run zero-shot NLP classification on the context snippet
        - Run wallet ML classification (if model loaded)
        - Save everything to DB
    
    ALWAYS runs zero-shot NLP classification, even if features fail.
    """
    try:
        if log_queue:
            log_queue.put(f"   🔎 Starting feature extraction for: {address}")

        # -----------------------------
        # STEP 1 — NLP classification (ALWAYS DO THIS FIRST)
        # -----------------------------
        try:
            nlp_label = classify_snippet_zero_shot(context_snippet)
            if log_queue:
                log_queue.put(f"   🧠 NLP label: {nlp_label}")
        except Exception as e:
            nlp_label = "unknown"
            if log_queue:
                log_queue.put(f"   ⚠️ Zero-shot NLP failed: {e}")

        # -----------------------------
        # STEP 2 — Blockchain feature extraction
        # -----------------------------
        if fetch_blockcypher_full is None or compute_wallet_features_blockcypher is None:
            msg = "Feature extractor not available, skipping wallet features."
            if log_queue:
                log_queue.put(f"   ⚠️ {msg}")

            with _features_lock:
                address_features[address] = {}

            # Save DB row with NLP label, mark wallet as licit (default)
            try:
                save_address_to_db(
                    session_id, url, depth, title, address,
                    context_snippet, is_illicit=0, nlp_label=nlp_label
                )
            except Exception:
                pass

            return

        # -----------------------------
        # STEP 3 — Fetch from BlockCypher
        # -----------------------------
        bc_json = None
        try:
            bc_json = fetch_blockcypher_full(
                address,
                token=BLOCKCYPHER_TOKEN,
                limit=50,
                sleep_sec=0.5
            )
        except Exception as e:
            if log_queue:
                log_queue.put(f"   ⚠️ Fetch failed (1st try) for {address}: {e}")

            # Retry once
            try:
                time.sleep(1)
                bc_json = fetch_blockcypher_full(
                    address,
                    token=BLOCKCYPHER_TOKEN,
                    limit=50,
                    sleep_sec=0.5
                )
            except Exception as e2:
                if log_queue:
                    log_queue.put(f"   ⚠️ Fetch failed (2nd try) for {address}: {e2}")

                # Mark empty
                with _features_lock:
                    address_features[address] = {}

                # Save with NLP label, wallet=licit
                save_address_to_db(
                    session_id, url, depth, title, address,
                    context_snippet, is_illicit=0, nlp_label=nlp_label
                )
                return

        # -----------------------------
        # STEP 4 — Compute blockchain features
        # -----------------------------
        feats = {}
        try:
            feats = compute_wallet_features_blockcypher(bc_json, address)
        except Exception as e:
            if log_queue:
                log_queue.put(f"   ⚠️ Feature computation failed: {e}")
            feats = {}

        with _features_lock:
            address_features[address] = feats

        if log_queue:
            if feats:
                log_queue.put(f"   ✅ Features computed (n={len(feats)})")
            else:
                log_queue.put(f"   ⚠️ No blockchain features extracted.")

        # -----------------------------
        # STEP 5 — Wallet ML model inference
        # -----------------------------
        is_illicit = 0  # default (licit)

        try:
            # classify_features_and_save will:
            #   - scale features
            #   - predict using wallet model
            #   - save to DB (BUT we override it to NOT save)
            # So instead we directly run the inference logic here.
            if ARTIFACTS["scalers"] and ARTIFACTS["feature_cols"] and ARTIFACTS["model"]:
                scalers = ARTIFACTS["scalers"]
                feature_cols = ARTIFACTS["feature_cols"]
                model = ARTIFACTS["model"]

                # Build row in consistent order
                row = []
                for col in feature_cols:
                    if col in feats:
                        row.append(feats[col])
                    else:
                        row.append(0.0)  # fallback for missing features

                # Scale
                X_new = pd.DataFrame([row], columns=feature_cols).astype(float)

                for col in feature_cols:
                    try:
                        scaler = scalers[col]
                        X_new[col] = scaler.transform([[X_new[col].values[0]]])[0][0]
                    except:
                        pass

                pred = model.predict(X_new.values)[0]
                # Convert: model(0=illicit,1=licit) → DB(1=illicit,0=licit)
                is_illicit = 0 if int(pred) == 0 else 1

                if log_queue:
                    log_queue.put(f"   🔍 Wallet ML predicted illicit={is_illicit}")

        except Exception as e:
            if log_queue:
                log_queue.put(f"   ⚠️ Wallet ML inference failed: {e}")

        # -----------------------------
        # STEP 6 — FINAL SAVE TO DB
        # -----------------------------
        save_address_to_db(
            session_id, url, depth, title, address,
            context_snippet, is_illicit=is_illicit, nlp_label=nlp_label
        )

        if log_queue:
            log_queue.put(f"   💾 Saved address with NLP label='{nlp_label}' and illicit={is_illicit}")

    except Exception as e:
        if log_queue:
            log_queue.put(f"   ❌ Unexpected worker error for {address}: {e}")
        else:
            print("Feature worker error:", e)

        with _features_lock:
            address_features[address] = {}

        # final fallback save
        try:
            save_address_to_db(
                session_id, url, depth, title, address,
                context_snippet, is_illicit=0, nlp_label="unknown"
            )
        except Exception:
            pass



def classify_features_and_save(session_id: str, url: str, depth: int, title: str, address: str, context_snippet: str, feats: dict, log_queue: Optional[Queue] = None):
    """
    Given computed feature dict `feats`, run scalers+model -> infer -> save row in DB with is_illicit.
    If artifacts missing / prediction fails, we save as licit (is_illicit=0) as a safe default.
    """
    try:
        if log_queue:
            log_queue.put(f"   🔬 Running inference for {address}")

        # Ensure artifacts loaded
        if ARTIFACTS["scalers"] is None or ARTIFACTS["feature_cols"] is None or ARTIFACTS["model"] is None:
            if log_queue:
                log_queue.put("   ⚠️ Inference artifacts missing — saving as licit by default.")
            save_address_to_db(session_id, url, depth, title, address, context_snippet, is_illicit=0)
            return

        scalers = ARTIFACTS["scalers"]
        feature_cols = ARTIFACTS["feature_cols"]
        model = ARTIFACTS["model"]

        # Build ordered row according to feature_cols
        row = []
        missing = []
        for col in feature_cols:
            if col in feats:
                row.append(feats[col])
            else:
                alt1 = col.replace(" ", "_")
                alt2 = col.replace("_", " ")
                if alt1 in feats:
                    row.append(feats[alt1])
                elif alt2 in feats:
                    row.append(feats[alt2])
                else:
                    # fallback: try alnum match
                    simple_col = "".join(ch for ch in col if ch.isalnum()).lower()
                    matched = False
                    for k in feats.keys():
                        if "".join(ch for ch in k if ch.isalnum()).lower() == simple_col:
                            row.append(feats[k])
                            matched = True
                            break
                    if not matched:
                        row.append(0.0)
                        missing.append(col)

        if log_queue and missing:
            log_queue.put(f"   ⚠️ Missing features filled with 0: {missing}")

        # Create DataFrame and apply scalers column-wise
        X_new = pd.DataFrame([row], columns=feature_cols).astype(float)
        for col in feature_cols:
            try:
                scaler = scalers[col]
                # scaler expects 2D array; feed single value
                X_new[col] = scaler.transform([[X_new[col].values[0]]])[0][0]
            except Exception as ex:
                # If scaler fails, leave raw value and warn
                if log_queue:
                    log_queue.put(f"   ⚠️ Scaler apply failed for {col}: {ex}")
                # continue with raw value

        # Predict
        try:
            pred = model.predict(X_new.values)[0]
            probs = model.predict_proba(X_new.values)[0] if hasattr(model, "predict_proba") else None
        except Exception as ex:
            if log_queue:
                log_queue.put(f"   ❌ Model prediction failed for {address}: {ex}")
            # fallback: save licit
            save_address_to_db(session_id, url, depth, title, address, context_snippet, is_illicit=0)
            return

        # Convert model prediction -> DB is_illicit convention
        # NOTE: your training mapping earlier was 0=illicit,1=licit. DB uses 1=illicit,0=licit.
        # So convert accordingly. If your model mapping later changes, update here.
        is_illicit_db = 1 if int(pred) == 0 else 0

        save_address_to_db(session_id, url, depth, title, address, context_snippet, is_illicit=is_illicit_db)

        if log_queue:
            log_queue.put(f"   ✅ Predicted model={pred} (probs={probs}); saved is_illicit={is_illicit_db}")

    except Exception as e:
        if log_queue:
            log_queue.put(f"   ❌ Unexpected error in classification: {e}")
        # best-effort save as licit
        try:
            save_address_to_db(session_id, url, depth, title, address, context_snippet, is_illicit=0)
        except Exception:
            pass




def save_address_to_db(session_id: str, url: str, depth: int, title: str, address: str, context_snippet: str, is_illicit: int = 0, nlp_label: str = None):
    """Thread-safe insert of a discovered address and its context into SQLite.

    is_illicit: 0 -> licit (default), 1 -> illicit
    """
    try:
        with _db_lock:
            cur = _db_conn.cursor()
            cur.execute("""
                INSERT INTO extracted_addresses 
                (session_id, url, depth, title, address, context_snippet, saved_at, is_illicit, nlp_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                url,
                depth,
                title,
                address,
                context_snippet,
                datetime.datetime.utcnow().isoformat(),
                int(is_illicit),
                nlp_label
            ))
            _db_conn.commit()
    except Exception as e:
        # do not raise — log to console (or to the crawl log) in production
        print(f"Error saving address to DB: {e}", traceback.format_exc())

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
    """Breadth-first crawl that pushes log lines into log_queue. Checks stop_event to exit early.

    When a Bitcoin address is found on a page, this function will start a background
    thread to compute features (compute_and_store_features) and store them in-memory.
    The crawler continues without waiting for feature extraction to finish.
    """
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

            # Find BTC addresses and kick off feature extraction in background
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
                    log_queue.put(f"   ⏳ Feature extraction started for: {addr}")
                except Exception as e:
                    log_queue.put(f"   ❌ Failed to start feature worker for {addr}: {e}")
                    # Mark as empty to avoid indefinite "not seen" state
                    try:
                        with _features_lock:
                            address_features[addr] = {}
                    except Exception:
                        pass

            # Add links to queue (basic same-origin-agnostic logic as before)
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

    # start crawler in background thread
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
                "saved_at": r[7],
                "is_illicit": int(r[8]) if r[8] is not None else 0,
                "nlp_label": r[9] if r[9] is not None else None
            })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
