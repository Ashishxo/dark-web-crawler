import time
import sys
import joblib
import numpy as np
import pandas as pd
from blockchain_info_extractor import fetch_all_transactions, compute_features

# ─── CONFIG ───
SAMPLE_SIZE_PER_CLASS = 15
SLEEP_BETWEEN_ADDRESSES = 5  # blockchain.info is more generous but still be polite

# ─── LOAD MODEL ARTIFACTS ───
print("Loading model artifacts...")
try:
    scalers = joblib.load("wallet_scalers.pkl")
    feature_cols = joblib.load("feature_order.pkl")
    model = joblib.load("random_forest_model.pkl")
    print(f"  Model loaded. Feature count: {len(feature_cols)}")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  Make sure wallet_scalers.pkl, feature_order.pkl, random_forest_model.pkl exist.")
    sys.exit(1)

# ─── LOAD DATASET ───
print("\nLoading Elliptic++ dataset...")
df = pd.read_csv("wallets_features_classes_combined.csv")
df = df.drop(columns=["Time step"]).drop_duplicates()

# Class 1 = illicit, Class 2 = licit
illicit_addresses = df[df["class"] == 1]["address"].unique()
licit_addresses = df[df["class"] == 2]["address"].unique()

print(f"  Unique illicit addresses: {len(illicit_addresses)}")
print(f"  Unique licit addresses:   {len(licit_addresses)}")

# ─── SAMPLE ───
rng = np.random.RandomState(42)
illicit_sample = rng.choice(illicit_addresses, size=min(SAMPLE_SIZE_PER_CLASS, len(illicit_addresses)), replace=False)
licit_sample = rng.choice(licit_addresses, size=min(SAMPLE_SIZE_PER_CLASS, len(licit_addresses)), replace=False)

# Model labels: 0 = licit, 1 = illicit
test_addresses = []
for addr in illicit_sample:
    test_addresses.append((addr, "illicit", 1))
for addr in licit_sample:
    test_addresses.append((addr, "licit", 0))

print(f"\nTesting {len(test_addresses)} addresses ({SAMPLE_SIZE_PER_CLASS} illicit + {SAMPLE_SIZE_PER_CLASS} licit)")
print("=" * 80)


# ─── INFERENCE FUNCTION ───
def run_inference(feats_dict):
    """
    Scale features and run RF model.
    Returns (prediction, probabilities).
    prediction: 0 = licit, 1 = illicit
    """
    row = []
    for col in feature_cols:
        val = feats_dict.get(col, None)
        if val is None:
            # Try space/underscore variant
            alt = col.replace("_", " ") if "_" in col else col.replace(" ", "_")
            val = feats_dict.get(alt, 0.0)
        row.append(float(val) if val is not None else 0.0)

    X = pd.DataFrame([row], columns=feature_cols).astype(float)

    # Apply per-column scalers
    for col in feature_cols:
        try:
            X[col] = scalers[col].transform([[X[col].values[0]]])[0][0]
        except:
            pass

    pred = model.predict(X.values)[0]
    probs = model.predict_proba(X.values)[0] if hasattr(model, "predict_proba") else None
    return int(pred), probs


# ─── RUN TESTS ───
results = []

for i, (address, true_label, true_model_label) in enumerate(test_addresses):
    print(f"\n[{i+1}/{len(test_addresses)}] {address}")
    print(f"  True: {true_label} (model class {true_model_label})")

    # Step 1: Fetch transactions
    try:
        api_data = fetch_all_transactions(address, limit=50, sleep_sec=2.0)
        n_txs = len(api_data.get("txs", []))
        print(f"  Transactions: {n_txs}")
    except Exception as e:
        print(f"  ❌ Fetch FAILED: {e}")
        results.append({
            "address": address, "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": None, "correct": None, "error": str(e)
        })
        time.sleep(SLEEP_BETWEEN_ADDRESSES)
        continue

    # Step 2: Compute features
    try:
        feats = compute_features(api_data, address)
        print(f"  Features: {len(feats)}")
    except Exception as e:
        print(f"  ❌ Feature computation FAILED: {e}")
        results.append({
            "address": address, "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": None, "correct": None, "error": str(e)
        })
        time.sleep(SLEEP_BETWEEN_ADDRESSES)
        continue

    # Step 3: Inference
    try:
        pred, probs = run_inference(feats)
        pred_label = "illicit" if pred == 1 else "licit"
        correct = (pred == true_model_label)

        print(f"  Predicted: {pred_label} (model class {pred})")
        if probs is not None:
            print(f"  Probabilities: licit={probs[0]:.3f}, illicit={probs[1]:.3f}")
        print(f"  {'✅ CORRECT' if correct else '❌ WRONG'}")

        results.append({
            "address": address, "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": pred, "predicted_label": pred_label,
            "correct": correct,
            "prob_licit": probs[0] if probs is not None else None,
            "prob_illicit": probs[1] if probs is not None else None,
            "error": None
        })
    except Exception as e:
        print(f"  ❌ Inference FAILED: {e}")
        results.append({
            "address": address, "true_label": true_label,
            "true_model_label": true_model_label,
            "predicted": None, "correct": None, "error": str(e)
        })

    time.sleep(SLEEP_BETWEEN_ADDRESSES)


# ─── SUMMARY ───
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

successful = [r for r in results if r["error"] is None]
failed = [r for r in results if r["error"] is not None]

print(f"\nTotal tested:    {len(results)}")
print(f"Successful:      {len(successful)}")
print(f"Failed (errors): {len(failed)}")

if successful:
    correct_count = sum(1 for r in successful if r["correct"])
    print(f"\nOverall accuracy: {correct_count}/{len(successful)} = {correct_count/len(successful)*100:.1f}%")

    # Illicit breakdown
    illicit_results = [r for r in successful if r["true_label"] == "illicit"]
    if illicit_results:
        illicit_correct = sum(1 for r in illicit_results if r["correct"])
        print(f"\nIllicit: {illicit_correct}/{len(illicit_results)} correct ({illicit_correct/len(illicit_results)*100:.1f}%)")
        for r in illicit_results:
            status = "✅" if r["correct"] else "❌"
            print(f"  {status} {r['address'][:20]}... → {r.get('predicted_label','?')} "
                  f"(P(illicit)={r.get('prob_illicit','?')})")

    # Licit breakdown
    licit_results = [r for r in successful if r["true_label"] == "licit"]
    if licit_results:
        licit_correct = sum(1 for r in licit_results if r["correct"])
        print(f"\nLicit: {licit_correct}/{len(licit_results)} correct ({licit_correct/len(licit_results)*100:.1f}%)")
        for r in licit_results:
            status = "✅" if r["correct"] else "❌"
            print(f"  {status} {r['address'][:20]}... → {r.get('predicted_label','?')} "
                  f"(P(illicit)={r.get('prob_illicit','?')})")

if failed:
    print(f"\nFailed addresses:")
    for r in failed:
        print(f"  {r['address']}: {r['error'][:80]}")

print("\nDone.")