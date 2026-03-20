import joblib
import numpy as np
import pandas as pd

# Load saved components
scalers = joblib.load("wallet_scalers.pkl")
feature_cols = joblib.load("feature_order.pkl")
model = joblib.load("random_forest_model.pkl")

# Your new feature vector (raw)
new_features = {
  "num_txs_as_sender": 1,
  "num_txs_as receiver": 1,
  "first_block_appeared_in": 393216,
  "last_block_appeared_in": 393218,
  "lifetime_in_blocks": 2,
  "total_txs": 2,
  "first_sent_block": 393218,
  "first_received_block": 393216,
  "num_timesteps_appeared_in": 1,
  "btc_transacted_total": 0.78291932,
  "btc_transacted_min": 0.39145966,
  "btc_transacted_max": 0.39145966,
  "btc_transacted_mean": 0.39145966,
  "btc_transacted_median": 0.39145966,
  "btc_sent_total": 0.39145966,
  "btc_sent_min": 0.39145966,
  "btc_sent_max": 0.39145966,
  "btc_sent_mean": 0.39145966,
  "btc_sent_median": 0.39145966,
  "btc_received_total": 0.39145966,
  "btc_received_min": 0.39145966,
  "btc_received_max": 0.39145966,
  "btc_received_mean": 0.39145966,
  "btc_received_median": 0.39145966,
  "fees_total": 0.00109687,
  "fees_min": 0.00109687,
  "fees_max": 0.00109687,
  "fees_mean": 0.00109687,
  "fees_median": 0.00109687,
  "fees_as_share_total": 0.0053565417187559,
  "fees_as_share_min": 0.002554541635273479,
  "fees_as_share_max": 0.002802000083482421,
  "fees_as_share_mean": 0.00267827085937795,
  "fees_as_share_median": 0.00267827085937795,
  "blocks_btwn_txs_total": 2,
  "blocks_btwn_txs_min": 2,
  "blocks_btwn_txs_max": 2,
  "blocks_btwn_txs_mean": 2.0,
  "blocks_btwn_txs_median": 2.0,
  "blocks_btwn_input_txs_total": 0,
  "blocks_btwn_input_txs_min": 0,
  "blocks_btwn_input_txs_max": 0,
  "blocks_btwn_input_txs_mean": 0.0,
  "blocks_btwn_input_txs_median": 0.0,
  "blocks_btwn_output_txs_total": 0,
  "blocks_btwn_output_txs_min": 0,
  "blocks_btwn_output_txs_max": 0,
  "blocks_btwn_output_txs_mean": 0.0,
  "blocks_btwn_output_txs_median": 0.0,
  "num_addr_transacted_multiple": 0,
  "transacted_w_address_total": 24,
  "transacted_w_address_min": 1,
  "transacted_w_address_max": 1,
  "transacted_w_address_mean": 1.0,
  "transacted_w_address_median": 1.0
}

# Build DataFrame in correct order
row = []
for col in feature_cols:
    row.append(new_features.get(col, 0.0))  # default 0.0 for safety

X_new = pd.DataFrame([row], columns=feature_cols).astype(float)

# Apply scalers
for col in feature_cols:
    scaler = scalers[col]
    X_new[col] = scaler.transform([[X_new[col].values[0]]])[0][0]

# Predict
pred = model.predict(X_new.values)[0]
probs = model.predict_proba(X_new.values)[0]

print("Prediction:", pred)  # 0 = licit, 1 = illicit
print("Prob licit:", probs[0])
print("Prob illicit:", probs[1])
