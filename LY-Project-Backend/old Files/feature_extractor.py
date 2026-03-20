# blockcypher_wallet_extractor.py (with progress bars)
import time
import requests
import math
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional
from tqdm import tqdm   # <-- NEW

# -----------------------
# Helper: sats -> BTC
# -----------------------
def _sats_to_btc(x):
    try:
        return float(x) / 1e8
    except:
        return 0.0


# -----------------------
# Fetch all txs via BlockCypher /full endpoint (paged) WITH PROGRESS BAR
# -----------------------
def fetch_blockcypher_full(address: str, token: Optional[str] = None,
                           limit: int = 50, sleep_sec: float = 1.0) -> Dict[str, Any]:
    """
    Fetch full transaction history with progress bar.
    Pages using 'before=hash' until hasMore=False.
    """
    base = "https://api.blockcypher.com/v1/btc/main"
    url = f"{base}/addrs/{address}/full"
    params = {"limit": limit}
    if token:
        params["token"] = token

    all_txs: List[Dict[str,Any]] = []
    page = 0
    last_hash = None

    print(f"\n🔄 Fetching full history for {address} (BlockCypher)...")

    # We don't know number of pages ahead → dynamic tqdm
    with tqdm(unit="page", desc="Downloading pages") as pbar:
        while True:
            if last_hash:
                params["before"] = last_hash

            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"BlockCypher fetch error status={resp.status_code} body={resp.text}"
                )

            j = resp.json()
            txs = j.get("txs") or []
            all_txs.extend(txs)

            # progress update
            pbar.update(1)

            # pagination control
            has_more = j.get("hasMore", False)
            page += 1

            if not has_more:
                break

            if txs:
                last_hash = txs[-1].get("hash") or txs[-1].get("txid")
                if not last_hash:
                    break
            else:
                break

            time.sleep(sleep_sec)

    print(f"📦 Total transactions fetched: {len(all_txs)}\n")
    return {"address": address, "txs": all_txs, "fetched_pages": page}


# -----------------------
# Compute wallet features (with progress bar)
# -----------------------
def compute_wallet_features_blockcypher(blockcypher_json: Dict[str,Any],
                                        wallet_address: str) -> Dict[str, Any]:
    """
    Compute Elliptic++ wallet features with a progress bar for transaction scanning.
    """
    txs = blockcypher_json.get("txs", [])
    if not isinstance(txs, list):
        txs = []

    print(f"🧮 Computing features for {wallet_address}...")
    time.sleep(0.2)

    # Prepare lists
    block_heights = []
    sent_amounts = []
    recv_amounts = []
    transacted_amounts = []
    fees_list = []
    tx_timestamps = []
    tx_is_sender = []
    tx_is_receiver = []
    counterparty_per_tx = []

    # Loop with tqdm
    for tx in tqdm(txs, desc="Processing TXs", unit="tx"):
        bh = tx.get("block_height", None)
        block_heights.append(bh if bh is not None else np.nan)

        is_sender = False
        is_receiver = False
        sent_val = 0.0
        recv_val = 0.0

        # inputs
        for vin in tx.get("inputs", []) or []:
            in_addrs = vin.get("addresses") or []
            if wallet_address in in_addrs:
                is_sender = True
                sent_val += _sats_to_btc(vin.get("output_value", 0))

        # outputs
        for vout in tx.get("outputs", []) or []:
            out_addrs = vout.get("addresses") or []
            if wallet_address in out_addrs:
                is_receiver = True
                recv_val += _sats_to_btc(vout.get("value", 0))

        sent_amounts.append(sent_val)
        recv_amounts.append(recv_val)
        transacted_amounts.append(sent_val + recv_val)
        tx_is_sender.append(is_sender)
        tx_is_receiver.append(is_receiver)

        fees_list.append(_sats_to_btc(tx.get("fees", 0)))

        ts_iso = tx.get("confirmed") or tx.get("received")
        tx_timestamps.append(ts_iso)

        # counterparties
        cp = set()
        for vin in tx.get("inputs", []) or []:
            for a in vin.get("addresses") or []:
                if a != wallet_address: cp.add(a)
        for vout in tx.get("outputs", []) or []:
            for a in vout.get("addresses") or []:
                if a != wallet_address: cp.add(a)
        counterparty_per_tx.append(cp)

    # Now convert to arrays and compute stats (same as your code)
    # ------------------------------- ARRAY CONVERSIONS -------------------------------
    bh_arr = np.array([np.nan if x is None else x for x in block_heights], dtype=float)
    sent_arr = np.array(sent_amounts, dtype=float)
    recv_arr = np.array(recev_amounts if (recev_amounts := recv_amounts) else recv_amounts, dtype=float)
    trans_arr = np.array(transacted_amounts, dtype=float)
    fees_arr = np.array(fees_list, dtype=float)

    # ------------------------------- HELPERS -------------------------------
    def _total(a): return float(np.nansum(a))
    def _min(a): return float(np.nanmin(a)) if np.any(~np.isnan(a)) else 0.0
    def _max(a): return float(np.nanmax(a)) if np.any(~np.isnan(a)) else 0.0
    def _mean(a): return float(np.nanmean(a)) if np.any(~np.isnan(a)) else 0.0
    def _median(a): return float(np.nanmedian(a)) if np.any(~np.isnan(a)) else 0.0

    # masks
    sender_mask = np.array(tx_is_sender, dtype=bool)
    receiver_mask = np.array(tx_is_receiver, dtype=bool)
    involved_mask = sender_mask | receiver_mask

    # ------------------------------- BASIC FEATURES -------------------------------
    num_txs_as_sender = int(sender_mask.sum())
    num_txs_as_receiver = int(receiver_mask.sum())
    total_txs = int(involved_mask.sum())

    valid_bh = bh_arr[~np.isnan(bh_arr)]
    if valid_bh.size:
        first_block = int(np.nanmin(valid_bh))
        last_block = int(np.nanmax(valid_bh))
        lifetime = last_block - first_block
    else:
        first_block = last_block = lifetime = 0

    sent_bh = np.array([b for b, s in zip(bh_arr, sender_mask) if not np.isnan(b) and s])
    recv_bh = np.array([b for b, r in zip(bh_arr, receiver_mask) if not np.isnan(b) and r])

    first_sent_block = int(sent_bh.min()) if sent_bh.size else 0
    first_received_block = int(recv_bh.min()) if recv_bh.size else 0

    # timesteps appeared
    dates = set()
    for ts in tx_timestamps:
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                dates.add(dt.date().isoformat())
            except:
                dates.add(ts[:10])
    num_timesteps = len(dates)

    # ------------------------------- AMOUNTS & FEES -------------------------------
    # transacted
    if involved_mask.sum() > 0:
        inv = trans_arr[involved_mask]
        btc_trans_total = _total(inv)
        btc_trans_min = _min(inv)
        btc_trans_max = _max(inv)
        btc_trans_mean = _mean(inv)
        btc_trans_median = _median(inv)
    else:
        btc_trans_total = btc_trans_min = btc_trans_max = btc_trans_mean = btc_trans_median = 0.0

    # sent
    if sender_mask.sum() > 0:
        s_vals = sent_arr[sender_mask]
        btc_sent_total = _total(s_vals)
        btc_sent_min = _min(s_vals)
        btc_sent_max = _max(s_vals)
        btc_sent_mean = _mean(s_vals)
        btc_sent_median = _median(s_vals)
    else:
        btc_sent_total = btc_sent_min = btc_sent_max = btc_sent_mean = btc_sent_median = 0.0

    # received
    if receiver_mask.sum() > 0:
        r_vals = recv_arr[receiver_mask]
        btc_recv_total = _total(r_vals)
        btc_recv_min = _min(r_vals)
        btc_recv_max = _max(r_vals)
        btc_recv_mean = _mean(r_vals)
        btc_recv_median = _median(r_vals)
    else:
        btc_recv_total = btc_recv_min = btc_recv_max = btc_recv_mean = btc_recv_median = 0.0

    # fees
    if sender_mask.sum() > 0:
        f_vals = fees_arr[sender_mask]
        fees_total = _total(f_vals)
        fees_min = _min(f_vals)
        fees_max = _max(f_vals)
        fees_mean = _mean(f_vals)
        fees_median = _median(f_vals)
    else:
        fees_total = fees_min = fees_max = fees_mean = fees_median = 0.0

    # fees_as_share = fee / transacted
    shares = []
    for f, t in zip(fees_arr, trans_arr):
        if t > 0:
            shares.append(f / t)
    shares = np.array(shares)
    if shares.size:
        fas_total = float(shares.sum())
        fas_min = float(shares.min())
        fas_max = float(shares.max())
        fas_mean = float(shares.mean())
        fas_median = float(np.median(shares))
    else:
        fas_total = fas_min = fas_max = fas_mean = fas_median = 0.0

    # ------------------------------- BLOCK DIFFERENCES -------------------------------
    involved_bh = sorted([int(b) for b, inv in zip(bh_arr, involved_mask) if not np.isnan(b) and inv])
    if len(involved_bh) >= 2:
        dif = np.diff(involved_bh)
        bt_total = int(dif.sum())
        bt_min = int(dif.min())
        bt_max = int(dif.max())
        bt_mean = float(dif.mean())
        bt_median = float(np.median(dif))
    else:
        bt_total = bt_min = bt_max = 0
        bt_mean = bt_median = 0.0

    # sent-only diffs
    sbh = sorted([int(b) for b, s in zip(bh_arr, sender_mask) if not np.isnan(b) and s])
    if len(sbh) >= 2:
        ds = np.diff(sbh)
        b_i_total = int(ds.sum())
        b_i_min = int(ds.min())
        b_i_max = int(ds.max())
        b_i_mean = float(ds.mean())
        b_i_median = float(np.median(ds))
    else:
        b_i_total = b_i_min = b_i_max = 0
        b_i_mean = b_i_median = 0.0

    # recv-only diffs
    rbh = sorted([int(b) for b, r in zip(bh_arr, receiver_mask) if not np.isnan(b) and r])
    if len(rbh) >= 2:
        dr = np.diff(rbh)
        b_o_total = int(dr.sum())
        b_o_min = int(dr.min())
        b_o_max = int(dr.max())
        b_o_mean = float(dr.mean())
        b_o_median = float(np.median(dr))
    else:
        b_o_total = b_o_min = b_o_max = 0
        b_o_mean = b_o_median = 0.0

    # ------------------------------- COUNTERPARTY STATS -------------------------------
    cp_map = defaultdict(int)
    for cp_set in counterparty_per_tx:
        for cp in cp_set:
            cp_map[cp] += 1

    cp_counts = np.array(list(cp_map.values()), dtype=int)
    if cp_counts.size:
        num_multi = int(np.sum(cp_counts > 1))
        cp_total = int(cp_counts.sum())
        cp_min = int(cp_counts.min())
        cp_max = int(cp_counts.max())
        cp_mean = float(cp_counts.mean())
        cp_median = float(np.median(cp_counts))
    else:
        num_multi = cp_total = cp_min = cp_max = 0
        cp_mean = cp_median = 0.0

    # return feature dict
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


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    addr = "112FBwDQ21CYxX785HE8qnQwkoDusYsTxC"
    token = "51eaeb12a21b4a4f85082d5b7c86ec44"  # your BlockCypher token OR None
    data = fetch_blockcypher_full(addr, token=token)
    feats = compute_wallet_features_blockcypher(data, addr)

    import json
    print("\n📊 Features:")
    print(json.dumps(feats, indent=2))
