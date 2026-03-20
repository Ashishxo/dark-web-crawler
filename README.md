# Dark Web Crawler & Bitcoin Wallet Transaction Analysis

A tool that crawls dark web (.onion) sites via Tor, extracts Bitcoin wallet addresses, and classifies them as illicit or licit using NLP and machine learning.


## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Tor** — must be running as a service with SOCKS proxy on port 9050

### Setting up Tor

Install the Tor Expert Bundle from [torproject.org](https://www.torproject.org/download/tor/).

Start the Tor service:
```bash
tor
```

Verify it's running — SOCKS proxy should be listening on `127.0.0.1:9050`.



## Backend Setup

```bash
cd LY-Project-Backend
```

### Install dependencies

```bash
pip install fastapi uvicorn requests[socks] beautifulsoup4 stem
pip install pandas numpy scikit-learn joblib
pip install transformers torch
```

### Required files in the backend directory

These files are needed for ML inference (generated from the training notebook):

- `wallet_scalers.pkl` — per-feature MinMaxScalers
- `feature_order.pkl` — ordered list of 55 feature column names
- `random_forest_model.pkl` — trained Random Forest model


### Start the backend

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

##### Note: On first run, the NLP model (`roberta-large-mnli`) will be downloaded (~1.3 GB). This is a one-time download. Subsequent starts will load from cache in 20-40 seconds.

You should see:
```
Loading zero-shot classifier… this may take 20-40 seconds
Zero-shot classifier loaded.
Loaded blockchain.info feature extractor.
Loaded wallet_scalers.pkl
Loaded feature_order.pkl
Loaded random_forest_model.pkl
Application startup complete.
```

## Frontend Setup

```bash
cd LY-Project-Frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:5173`.

## Usage

1. **Make sure Tor is running** before starting the backend.

2. Open the frontend in your browser (`http://localhost:5173`).

3. **Dark Web Spider** — Enter a keyword (e.g., "bitcoin marketplace") or paste a .onion URL. Set the crawl depth (1-3). Click "Start Crawling". A new tab opens showing live crawl logs.

4. **Extracted Addresses** — View all Bitcoin addresses found across crawl sessions, along with their NLP labels and illicit/licit classification.

5. **Graph Visualizer** — Click any address to see its transaction graph showing counterparty relationships.

6. **Malicious Wallets** — Filtered view showing only addresses classified as illicit. Supports CSV export.

## Project Structure

```
├── LY-Project-Backend/
│   ├── server.py                      # FastAPI backend + crawler
│   ├── blockchain_info_extractor.py   # Feature extraction (blockchain.info API)
│   ├── nlp_model.py                   # Zero-shot NLP classifier setup
│   ├── wallet_scalers.pkl             # Saved scalers for inference
│   ├── feature_order.pkl              # Feature column order
│   ├── random_forest_model.pkl        # Trained Random Forest model
│   └── extracted_addresses.db         # SQLite database (created at runtime)
│
├── LY-Project-Frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── Pages/
│   │       ├── DarkWebSpider.jsx      # Crawl launcher
│   │       ├── CrawlPage.jsx          # Live crawl log viewer
│   │       ├── Addresses.jsx          # All extracted addresses
│   │       ├── MaliciousWallet.jsx    # Illicit wallets filter
│   │       └── Graph.jsx              # Wallet transaction graph
│   └── package.json
│
└── Wallet_Classification_ML.ipynb     # Model training notebook
```

## References

- **Elliptic++ Dataset**: Elmougy & Liu (2023), "Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics", KDD '23.
- **Zero-shot NLP**: RoBERTa-large-MNLI via HuggingFace Transformers.
- **Blockchain API**: blockchain.info/rawaddr endpoint (no API key required).