import time
import requests
import numpy as np
import threading
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, List, Optional

# lock to prevent multiple threads from hitting blockchain.info simultaneously
_api_lock = threading.Lock()


def _sats_to_btc(x):
    """Convert satoshis to BTC."""
    try:
        return float(x) / 1e8
    except:
        return 0.0


def _safe_stats(arr):
    """Return (total, min, max, mean, median) for a numpy array, or zeros if empty."""
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.sum(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.median(arr)),
    )


# ─── Fetch all transactions via blockchain.info ───

def fetch_all_transactions(address: str, limit: int = 50, sleep_sec: float = 2.0) -> Dict[str, Any]:
    
    #Returns dict with keys: 'address', 'n_tx', 'txs', 'fetched_pages'

    base_url = f"https://blockchain.info/rawaddr/{address}"
    all_txs = []
    offset = 0
    page = 0
    total_tx = None

    while True:
        params = {"limit": limit, "offset": offset}

        # Only one thread can hit the API at a time
        with _api_lock:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "unknown")
                print(f"  ⚠️ Rate limited for {address}. Retry-After: {retry_after}")
                print(f"     Headers: {dict(resp.headers)}")
                wait = int(retry_after) if retry_after.isdigit() else 30
                time.sleep(wait)
                resp = requests.get(base_url, params=params, timeout=30)
            time.sleep(sleep_sec)

        if resp.status_code != 200:
            raise RuntimeError(
                f"blockchain.info error {resp.status_code}: {resp.text[:200]}\n"
                f"Headers: {dict(resp.headers)}"
            )

        j = resp.json()

        if total_tx is None:
            total_tx = j.get("n_tx", 0)

        txs = j.get("txs", [])
        if not txs:
            break

        all_txs.extend(txs)
        page += 1

        if len(all_txs) >= total_tx:
            break

        offset += limit

    return {
        "address": address,
        "n_tx": total_tx,
        "txs": all_txs,
        "fetched_pages": page,
    }


# =Compute 55 features

def compute_features(api_data: Dict[str, Any], wallet_address: str) -> Dict[str, Any]:
    """
    Compute all 55 Elliptic++ wallet features from blockchain.info transaction data.
    
    blockchain.info transaction format:
        tx['inputs'][i]['prev_out']['addr']   - input address
        tx['inputs'][i]['prev_out']['value']  - input value (satoshis)
        tx['out'][i]['addr']                  - output address
        tx['out'][i]['value']                 - output value (satoshis)
        tx['fee']                             - fee (satoshis)
        tx['block_height']                    - block number
        tx['time']                            - unix timestamp
    """
    txs = api_data.get("txs", [])
    if not txs:
        # Return all zeros
        return _empty_features()

    # ─── Pass 1: Classify each transaction ───
    block_heights = []
    sent_amounts = []
    recv_amounts = []
    transacted_amounts = []
    fees_list = []
    tx_timestamps = []
    tx_is_sender = []
    tx_is_receiver = []
    counterparty_per_tx = []
    transacted_amounts_full_tx = []

    for tx in txs:
        # Block height
        bh = tx.get("block_height", None)
        # blockchain.info uses -1 or 0 for unconfirmed
        if bh is not None and bh > 0:
            block_heights.append(float(bh))
        else:
            block_heights.append(np.nan)

        is_sender = False
        is_receiver = False
        sent_val = 0.0
        recv_val = 0.0

        # Check inputs (sender side)
        for inp in tx.get("inputs", []):
            prev_out = inp.get("prev_out", {})
            addr = prev_out.get("addr", "")
            value = prev_out.get("value", 0)
            if addr == wallet_address:
                is_sender = True
                sent_val += _sats_to_btc(value)

        # Check outputs (receiver side)
        for out in tx.get("out", []):
            addr = out.get("addr", "")
            value = out.get("value", 0)
            if addr == wallet_address:
                is_receiver = True
                recv_val += _sats_to_btc(value)

        sent_amounts.append(sent_val)
        recv_amounts.append(recv_val)
        transacted_amounts.append(sent_val + recv_val)
        tx_is_sender.append(is_sender)
        tx_is_receiver.append(is_receiver)

        # Fee
        fee = tx.get("fee", 0)
        fees_list.append(_sats_to_btc(fee))

        # Total transaction value (sum of ALL outputs, not just wallet's)
        # Used for fees_as_share calculation
        total_tx_val = 0.0
        for out in tx.get("out", []):
            total_tx_val += _sats_to_btc(out.get("value", 0))
        transacted_amounts_full_tx.append(total_tx_val)

        # Timestamp
        ts = tx.get("time", None)
        tx_timestamps.append(ts)

        # Counterparties: only count DIRECT counterparties
        # If wallet is receiver: counterparties are the input addresses (who sent TO wallet)
        # If wallet is sender: counterparties are the output addresses (who wallet sent TO)
        # NOT all addresses in the transaction
        cp = set()
        if is_receiver:
            for inp in tx.get("inputs", []):
                addr = inp.get("prev_out", {}).get("addr", "")
                if addr and addr != wallet_address:
                    cp.add(addr)
        if is_sender:
            for out in tx.get("out", []):
                addr = out.get("addr", "")
                if addr and addr != wallet_address:
                    cp.add(addr)
        counterparty_per_tx.append(cp)

    # ─── Convert to arrays ───
    bh_arr = np.array(block_heights, dtype=float)
    sent_arr = np.array(sent_amounts, dtype=float)
    recv_arr = np.array(recv_amounts, dtype=float)
    trans_arr = np.array(transacted_amounts, dtype=float)
    fees_arr = np.array(fees_list, dtype=float)
    sender_mask = np.array(tx_is_sender, dtype=bool)
    receiver_mask = np.array(tx_is_receiver, dtype=bool)
    involved_mask = sender_mask | receiver_mask

    # ─── Basic features ───
    num_txs_as_sender = int(sender_mask.sum())
    num_txs_as_receiver = int(receiver_mask.sum())
    total_txs = int(involved_mask.sum())

    valid_bh = bh_arr[~np.isnan(bh_arr)]
    if valid_bh.size:
        first_block = int(np.min(valid_bh))
        last_block = int(np.max(valid_bh))
        lifetime = last_block - first_block
    else:
        first_block = last_block = lifetime = 0

    sent_bh = np.array([b for b, s in zip(bh_arr, sender_mask) if not np.isnan(b) and s])
    recv_bh = np.array([b for b, r in zip(bh_arr, receiver_mask) if not np.isnan(b) and r])

    first_sent_block = int(sent_bh.min()) if sent_bh.size else 0
    first_received_block = int(recv_bh.min()) if recv_bh.size else 0

    # Timesteps appeared (unique dates)
    dates = set()
    for ts in tx_timestamps:
        if ts:
            try:
                dt = datetime.utcfromtimestamp(ts)
                dates.add(dt.date().isoformat())
            except:
                pass
    num_timesteps = len(dates)

    # ─── Amount stats ───
    if involved_mask.sum() > 0:
        inv = trans_arr[involved_mask]
        btc_trans_total, btc_trans_min, btc_trans_max, btc_trans_mean, btc_trans_median = _safe_stats(inv)
    else:
        btc_trans_total = btc_trans_min = btc_trans_max = btc_trans_mean = btc_trans_median = 0.0

    # Sent stats: computed over ALL involved transactions (0 for txs where wallet didn't send)
    if involved_mask.sum() > 0:
        s_vals = sent_arr[involved_mask]  # includes 0s for receive-only txs
        btc_sent_total, btc_sent_min, btc_sent_max, btc_sent_mean, btc_sent_median = _safe_stats(s_vals)
    else:
        btc_sent_total = btc_sent_min = btc_sent_max = btc_sent_mean = btc_sent_median = 0.0

    # Received stats: computed over ALL involved transactions (0 for txs where wallet didn't receive)
    if involved_mask.sum() > 0:
        r_vals = recv_arr[involved_mask]  # includes 0s for send-only txs
        btc_recv_total, btc_recv_min, btc_recv_max, btc_recv_mean, btc_recv_median = _safe_stats(r_vals)
    else:
        btc_recv_total = btc_recv_min = btc_recv_max = btc_recv_mean = btc_recv_median = 0.0

    # Fee stats: computed over ALL involved transactions (not just sends)
    if involved_mask.sum() > 0:
        f_vals = fees_arr[involved_mask]
        fees_total, fees_min, fees_max, fees_mean, fees_median = _safe_stats(f_vals)
    else:
        fees_total = fees_min = fees_max = fees_mean = fees_median = 0.0

    # Fees as share: fee / total_transaction_value (sum of all outputs in the tx)
    # Elliptic++ uses the full tx value as denominator, not just the wallet's portion
    shares = []
    for f, tv in zip(fees_arr, transacted_amounts_full_tx):
        if tv > 0:
            shares.append(f / tv)
    shares = np.array(shares)
    fas_total, fas_min, fas_max, fas_mean, fas_median = _safe_stats(shares)

    # ─── Block difference stats ───
    involved_bh = sorted([int(b) for b, inv in zip(bh_arr, involved_mask) if not np.isnan(b) and inv])
    if len(involved_bh) >= 2:
        dif = np.diff(involved_bh)
        bt_total, bt_min, bt_max, bt_mean, bt_median = int(dif.sum()), int(dif.min()), int(dif.max()), float(dif.mean()), float(np.median(dif))
    else:
        bt_total = bt_min = bt_max = 0
        bt_mean = bt_median = 0.0

    sbh = sorted([int(b) for b, s in zip(bh_arr, sender_mask) if not np.isnan(b) and s])
    if len(sbh) >= 2:
        ds = np.diff(sbh)
        b_i_total, b_i_min, b_i_max, b_i_mean, b_i_median = int(ds.sum()), int(ds.min()), int(ds.max()), float(ds.mean()), float(np.median(ds))
    else:
        b_i_total = b_i_min = b_i_max = 0
        b_i_mean = b_i_median = 0.0

    rbh = sorted([int(b) for b, r in zip(bh_arr, receiver_mask) if not np.isnan(b) and r])
    if len(rbh) >= 2:
        dr = np.diff(rbh)
        b_o_total, b_o_min, b_o_max, b_o_mean, b_o_median = int(dr.sum()), int(dr.min()), int(dr.max()), float(dr.mean()), float(np.median(dr))
    else:
        b_o_total = b_o_min = b_o_max = 0
        b_o_mean = b_o_median = 0.0

    # ─── Counterparty stats ───
    cp_map = defaultdict(int)
    for cp_set in counterparty_per_tx:
        for cp in cp_set:
            cp_map[cp] += 1

    cp_counts = np.array(list(cp_map.values()), dtype=int)
    if cp_counts.size:
        num_multi = int(np.sum(cp_counts > 1))
        cp_total, cp_min, cp_max, cp_mean, cp_median = _safe_stats(cp_counts.astype(float))
        cp_total = int(cp_total)
        cp_min = int(cp_min)
        cp_max = int(cp_max)
    else:
        num_multi = cp_total = cp_min = cp_max = 0
        cp_mean = cp_median = 0.0

    # ─── Return feature dict ───
    # Note: 'num_txs_as receiver' has a SPACE not underscore — matches Elliptic++ CSV
    return {
        'num_txs_as_sender': num_txs_as_sender,
        'num_txs_as receiver': num_txs_as_receiver,
        'first_block_appeared_in': first_block,
        'last_block_appeared_in': last_block,
        'lifetime_in_blocks': lifetime,
        'total_txs': total_txs,
        'first_sent_block': first_sent_block,
        'first_received_block': first_received_block,
        'num_timesteps_appeared_in': num_timesteps,
        'btc_transacted_total': btc_trans_total,
        'btc_transacted_min': btc_trans_min,
        'btc_transacted_max': btc_trans_max,
        'btc_transacted_mean': btc_trans_mean,
        'btc_transacted_median': btc_trans_median,
        'btc_sent_total': btc_sent_total,
        'btc_sent_min': btc_sent_min,
        'btc_sent_max': btc_sent_max,
        'btc_sent_mean': btc_sent_mean,
        'btc_sent_median': btc_sent_median,
        'btc_received_total': btc_recv_total,
        'btc_received_min': btc_recv_min,
        'btc_received_max': btc_recv_max,
        'btc_received_mean': btc_recv_mean,
        'btc_received_median': btc_recv_median,
        'fees_total': fees_total,
        'fees_min': fees_min,
        'fees_max': fees_max,
        'fees_mean': fees_mean,
        'fees_median': fees_median,
        'fees_as_share_total': fas_total,
        'fees_as_share_min': fas_min,
        'fees_as_share_max': fas_max,
        'fees_as_share_mean': fas_mean,
        'fees_as_share_median': fas_median,
        'blocks_btwn_txs_total': bt_total,
        'blocks_btwn_txs_min': bt_min,
        'blocks_btwn_txs_max': bt_max,
        'blocks_btwn_txs_mean': bt_mean,
        'blocks_btwn_txs_median': bt_median,
        'blocks_btwn_input_txs_total': b_i_total,
        'blocks_btwn_input_txs_min': b_i_min,
        'blocks_btwn_input_txs_max': b_i_max,
        'blocks_btwn_input_txs_mean': b_i_mean,
        'blocks_btwn_input_txs_median': b_i_median,
        'blocks_btwn_output_txs_total': b_o_total,
        'blocks_btwn_output_txs_min': b_o_min,
        'blocks_btwn_output_txs_max': b_o_max,
        'blocks_btwn_output_txs_mean': b_o_mean,
        'blocks_btwn_output_txs_median': b_o_median,
        'num_addr_transacted_multiple': num_multi,
        'transacted_w_address_total': cp_total,
        'transacted_w_address_min': cp_min,
        'transacted_w_address_max': cp_max,
        'transacted_w_address_mean': cp_mean,
        'transacted_w_address_median': cp_median,
    }


def _empty_features():
    """Return a dict with all 55 features set to 0."""
    keys = [
        'num_txs_as_sender', 'num_txs_as receiver', 'first_block_appeared_in',
        'last_block_appeared_in', 'lifetime_in_blocks', 'total_txs',
        'first_sent_block', 'first_received_block', 'num_timesteps_appeared_in',
        'btc_transacted_total', 'btc_transacted_min', 'btc_transacted_max',
        'btc_transacted_mean', 'btc_transacted_median',
        'btc_sent_total', 'btc_sent_min', 'btc_sent_max', 'btc_sent_mean', 'btc_sent_median',
        'btc_received_total', 'btc_received_min', 'btc_received_max',
        'btc_received_mean', 'btc_received_median',
        'fees_total', 'fees_min', 'fees_max', 'fees_mean', 'fees_median',
        'fees_as_share_total', 'fees_as_share_min', 'fees_as_share_max',
        'fees_as_share_mean', 'fees_as_share_median',
        'blocks_btwn_txs_total', 'blocks_btwn_txs_min', 'blocks_btwn_txs_max',
        'blocks_btwn_txs_mean', 'blocks_btwn_txs_median',
        'blocks_btwn_input_txs_total', 'blocks_btwn_input_txs_min',
        'blocks_btwn_input_txs_max', 'blocks_btwn_input_txs_mean',
        'blocks_btwn_input_txs_median',
        'blocks_btwn_output_txs_total', 'blocks_btwn_output_txs_min',
        'blocks_btwn_output_txs_max', 'blocks_btwn_output_txs_mean',
        'blocks_btwn_output_txs_median',
        'num_addr_transacted_multiple',
        'transacted_w_address_total', 'transacted_w_address_min',
        'transacted_w_address_max', 'transacted_w_address_mean',
        'transacted_w_address_median',
    ]
    return {k: 0 for k in keys}


# ─── Comparison mode ───

if __name__ == "__main__":
    import pandas as pd
    import sys

    # Address to test — use first argument or default
    ADDRESS = sys.argv[1] if len(sys.argv) > 1 else "1GfUF5ePLp98FYF8LZ6VdMpsJUr41Hc7kB"

    # Load CSV
    print("Loading Elliptic++ CSV...")
    df = pd.read_csv("wallets_features_classes_combined.csv")
    df = df.drop(columns=["Time step"]).drop_duplicates()

    csv_row = df[df["address"] == ADDRESS]
    if csv_row.empty:
        print(f"Address {ADDRESS} not found in CSV!")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in ["address", "class"]]
    csv_features = csv_row[feature_cols].iloc[0].to_dict()

    print(f"Address: {ADDRESS}")
    print(f"Class: {csv_row['class'].values[0]} (1=illicit, 2=licit)")

    # Fetch and compute
    api_data = fetch_all_transactions(ADDRESS, limit=50, sleep_sec=1.0)
    api_features = compute_features(api_data, ADDRESS)

    # Compare
    print()
    print("=" * 105)
    print(f"{'FEATURE':<40} {'CSV VALUE':>15} {'API VALUE':>15} {'MATCH?':>8} {'DIFF':>15} {'% OFF':>10}")
    print("=" * 105)

    match_count = 0
    mismatch_count = 0
    missing_count = 0

    for col in feature_cols:
        csv_val = csv_features.get(col, None)
        api_val = api_features.get(col, None)

        if api_val is None:
            status = "MISS"
            diff_str = "—"
            pct_str = "—"
            missing_count += 1
        elif csv_val is not None and abs(float(csv_val) - float(api_val)) < 0.001:
            status = "✅"
            diff_str = "0"
            pct_str = "0%"
            match_count += 1
        else:
            status = "❌"
            diff_val = float(api_val) - float(csv_val)
            diff_str = f"{diff_val:.4f}"
            if float(csv_val) != 0:
                pct_str = f"{abs(diff_val / float(csv_val)) * 100:.1f}%"
            else:
                pct_str = "inf" if diff_val != 0 else "0%"
            mismatch_count += 1

        csv_display = f"{float(csv_val):.4f}" if csv_val is not None else "—"
        api_display = f"{float(api_val):.4f}" if api_val is not None else "—"

        print(f"{col:<40} {csv_display:>15} {api_display:>15} {status:>8} {diff_str:>15} {pct_str:>10}")

    print("=" * 105)
    print(f"\n✅ Matched:  {match_count}/{len(feature_cols)}")
    print(f"❌ Mismatch: {mismatch_count}/{len(feature_cols)}")
    print(f"🔍 Missing:  {missing_count}/{len(feature_cols)}")