

import numpy as np
import pandas as pd
from feature_extractor import fetch_blockcypher_full, compute_wallet_features_blockcypher

# ─── Load dataset ───
df = pd.read_csv("wallets_features_classes_combined.csv")
df = df.drop(columns=["Time step"]).drop_duplicates()

# Use one of the addresses that successfully fetched earlier
ADDRESS = "12nhVzUgGBMdRMhxYMQWC5nq4rBewMzj9x"
BLOCKCYPHER_TOKEN = "51eaeb12a21b4a4f85082d5b7c86ec44"

# ─── Get CSV features ───
csv_row = df[df["address"] == ADDRESS]
if csv_row.empty:
    print(f"Address {ADDRESS} not found in CSV!")
    exit()

print(f"Address: {ADDRESS}")
print(f"Class in CSV: {csv_row['class'].values[0]} (1=illicit, 2=licit)")
print()

# Feature columns
feature_cols = [c for c in df.columns if c not in ["address", "class"]]
csv_features = csv_row[feature_cols].iloc[0].to_dict()

# ─── Fetch and compute from API ───
print("Fetching from BlockCypher...")
bc_json = fetch_blockcypher_full(ADDRESS, token=BLOCKCYPHER_TOKEN, limit=50, sleep_sec=0.5)
print(f"Transactions fetched: {len(bc_json.get('txs', []))}")
print()

print("Computing features...")
api_features = compute_wallet_features_blockcypher(bc_json, ADDRESS)
print()

# ─── Compare ───
print("=" * 100)
print(f"{'FEATURE':<40} {'CSV VALUE':>15} {'API VALUE':>15} {'MATCH?':>10} {'DIFF':>15}")
print("=" * 100)

match_count = 0
mismatch_count = 0
missing_count = 0

for col in feature_cols:
    csv_val = csv_features.get(col, None)
    
    # Try exact key match, then space/underscore variants
    api_val = api_features.get(col, None)
    if api_val is None:
        alt = col.replace("_", " ") if "_" in col else col.replace(" ", "_")
        api_val = api_features.get(alt, None)
    
    if api_val is None:
        status = "MISSING"
        diff = "—"
        missing_count += 1
    elif csv_val is not None and abs(float(csv_val) - float(api_val)) < 0.001:
        status = "✅"
        diff = "0"
        match_count += 1
    else:
        status = "❌"
        if csv_val is not None:
            diff = f"{float(api_val) - float(csv_val):.4f}"
        else:
            diff = "—"
        mismatch_count += 1
    
    csv_display = f"{float(csv_val):.4f}" if csv_val is not None else "—"
    api_display = f"{float(api_val):.4f}" if api_val is not None else "—"
    
    print(f"{col:<40} {csv_display:>15} {api_display:>15} {status:>10} {diff:>15}")

print("=" * 100)
print(f"\nMatched:  {match_count}")
print(f"Mismatch: {mismatch_count}")
print(f"Missing:  {missing_count}")
print(f"Total:    {len(feature_cols)}")