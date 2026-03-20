"""
validate_pipeline.py

Purpose: End-to-end validation of the feature extraction + inference pipeline.

Takes known illicit (class=1) and licit (class=2) addresses from the Elliptic++ dataset,
fetches their transactions from BlockCypher, computes features using the same
feature_extractor.py used in the server, runs inference using the saved model,
and checks how many are correctly classified.

Usage:
    python validate_pipeline.py

Requirements:
    - wallets_features_classes_combined.csv (Elliptic++ dataset)
    - wallet_scalers.pkl (saved scalers)
    - feature_order.pkl (saved feature column order)
    - random_forest_model.pkl (saved model)
    - feature_extractor.py (in same directory)

Note: BlockCypher free tier has rate limits (~200 requests/hour without token,
      ~2000/hour with token). This script tests a small sample.
"""

import time
import sys
import joblib
import numpy as np
import pandas as pd
from feature_extractor import fetch_blockcypher_full, compute_wallet_features_blockcypher

# ─── CONFIG ───
BLOCKCYPHER_TOKEN = "51eaeb12a21b4a4f85082d5b7c86ec44"  # or None for no token
SAMPLE_SIZE_PER_CLASS = 15  # how many addresses to test per class (illicit / licit)
SLEEP_BETWEEN_ADDRESSES = 2  # seconds between API calls to avoid rate limits

# ─── LOAD ARTIFACTS ───
print("Loading model artifacts...")
try:
    scalers = joblib.load("wallet_scalers.pkl")
    feature_cols = joblib.load("feature_order.pkl")
    model = joblib.load("random_forest_model.pkl")
    print(f"  Model loaded. Feature count: {len(feature_cols)}")
except Exception as e:
    print(f"  ERROR loading artifacts: {e}")
    print("  Make sure wallet_scalers.pkl, feature_order.pkl, and random_forest_model.pkl exist.")
    sys.exit(1)

# ─── LOAD DATASET ───
print("\nLoading Elliptic++ dataset...")
df = pd.read_csv("wallets_features_classes_combined.csv")
df = df.drop(columns=["Time step"]).drop_duplicates()

# Class 1 = illicit, Class 2 = licit in Elliptic++
illicit_addresses = df[df["class"] == 1]["address"].unique()
licit_addresses = df[df["class"] == 2]["address"].unique()

print(f"  Total unique illicit addresses: {len(illicit_addresses)}")
print(f"  Total unique licit addresses:   {len(licit_addresses)}")

# ─── SAMPLE ───
# Randomly sample addresses to test
rng = np.random.RandomState(42)
illicit_sample = rng.choice(illicit_addresses, size=min(SAMPLE_SIZE_PER_CLASS, len(illicit_addresses)), replace=False)
licit_sample = rng.choice(licit_addresses, size=min(SAMPLE_SIZE_PER_CLASS, len(licit_addresses)), replace=False)

test_addresses = []
for addr in illicit_sample:
    test_addresses.append((addr, "illicit", 1))  # (address, true_label_name, true_model_label)
for addr in licit_sample:
    test_addresses.append((addr, "licit", 0))  # model: 0=licit, 1=illicit

print(f"\nTesting {len(test_addresses)} addresses ({SAMPLE_SIZE_PER_CLASS} illicit + {SAMPLE_SIZE_PER_CLASS} licit)")
print("=" * 80)


def run_inference(feats_dict):
    """
    Given a feature dict from compute_wallet_features_blockcypher,
    scale it and run the model. Returns (prediction, probabilities).
    
    prediction: 0=licit, 1=illicit
    """
    # Build row in feature_cols order
    row = []
    missing = []
    for col in feature_cols:
        if col in feats_dict:
            row.append(feats_dict[col])
        else:
            # Try common mismatches (space vs underscore)
            alt = col.replace("_", " ") if "_" in col else col.replace(" ", "_")
            if alt in feats_dict:
                row.append(feats_dict[alt])
            else:
                row.append(0.0)
                missing.append(col)

    if missing:
        print(f"    ⚠️  Missing features (filled with 0): {missing}")

    # Scale
    X_new = pd.DataFrame([row], columns=feature_cols).astype(float)
    for col in feature_cols:
        try:
            X_new[col] = scalers[col].transform([[X_new[col].values[0]]])[0][0]
        except:
            pass

    # Predict
    pred = model.predict(X_new.values)[0]
    probs = model.predict_proba(X_new.values)[0] if hasattr(model, "predict_proba") else None

    return int(pred), probs


# ─── RUN TESTS ───
results = []

for i, (address, true_label, true_model_label) in enumerate(test_addresses):
    print(f"\n[{i+1}/{len(test_addresses)}] Address: {address}")
    print(f"  True label: {true_label} (model class: {true_model_label})")

    # Step 1: Fetch from BlockCypher
    try:
        bc_json = fetch_blockcypher_full(address, token=BLOCKCYPHER_TOKEN, limit=50, sleep_sec=0.5)
        n_txs = len(bc_json.get("txs", []))
        print(f"  Fetched {n_txs} transactions")
    except Exception as e:
        print(f"  ❌ BlockCypher fetch FAILED: {e}")
        results.append({
            "address": address,
            "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": None,
            "correct": None,
            "error": str(e)
        })
        time.sleep(SLEEP_BETWEEN_ADDRESSES)
        continue

    # Step 2: Compute features
    try:
        feats = compute_wallet_features_blockcypher(bc_json, address)
        print(f"  Computed {len(feats)} features")
    except Exception as e:
        print(f"  ❌ Feature computation FAILED: {e}")
        results.append({
            "address": address,
            "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": None,
            "correct": None,
            "error": str(e)
        })
        time.sleep(SLEEP_BETWEEN_ADDRESSES)
        continue

    # Step 3: Run inference
    try:
        pred, probs = run_inference(feats)
        pred_label = "illicit" if pred == 1 else "licit"
        correct = (pred == true_model_label)

        print(f"  Prediction: {pred_label} (model class: {pred})")
        if probs is not None:
            print(f"  Probabilities: licit={probs[0]:.3f}, illicit={probs[1]:.3f}")
        print(f"  {'✅ CORRECT' if correct else '❌ WRONG'}")

        results.append({
            "address": address,
            "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": pred,
            "predicted_label": pred_label,
            "correct": correct,
            "prob_licit": probs[0] if probs is not None else None,
            "prob_illicit": probs[1] if probs is not None else None,
            "error": None
        })
    except Exception as e:
        print(f"  ❌ Inference FAILED: {e}")
        results.append({
            "address": address,
            "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": None,
            "correct": None,
            "error": str(e)
        })

    # Rate limit protection
    time.sleep(SLEEP_BETWEEN_ADDRESSES)

# ─── SUMMARY ───
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

successful = [r for r in results if r["error"] is None]
failed = [r for r in results if r["error"] is not None]

print(f"\nTotal addresses tested: {len(results)}")
print(f"Successful API + inference: {len(successful)}")
print(f"Failed (API/feature errors): {len(failed)}")

if successful:
    correct_count = sum(1 for r in successful if r["correct"])
    total_valid = len(successful)
    print(f"\nOverall accuracy: {correct_count}/{total_valid} = {correct_count/total_valid*100:.1f}%")

    # Break down by class
    illicit_results = [r for r in successful if r["true_label"] == "illicit"]
    licit_results = [r for r in successful if r["true_label"] == "licit"]

    if illicit_results:
        illicit_correct = sum(1 for r in illicit_results if r["correct"])
        print(f"\nIllicit addresses: {illicit_correct}/{len(illicit_results)} correct "
              f"({illicit_correct/len(illicit_results)*100:.1f}%)")
        # Show wrong ones
        wrong_illicit = [r for r in illicit_results if not r["correct"]]
        for r in wrong_illicit:
            print(f"  MISSED: {r['address']} → predicted licit "
                  f"(prob_illicit={r.get('prob_illicit', '?')})")

    if licit_results:
        licit_correct = sum(1 for r in licit_results if r["correct"])
        print(f"\nLicit addresses: {licit_correct}/{len(licit_results)} correct "
              f"({licit_correct/len(licit_results)*100:.1f}%)")
        wrong_licit = [r for r in licit_results if not r["correct"]]
        for r in wrong_licit:
            print(f"  FALSE ALARM: {r['address']} → predicted illicit "
                  f"(prob_illicit={r.get('prob_illicit', '?')})")

if failed:
    print(f"\nFailed addresses:")
    for r in failed:
        print(f"  {r['address']}: {r['error']}")

print("\nDone.")