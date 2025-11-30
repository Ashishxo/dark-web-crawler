// src/components/WalletGraph.jsx
import React, { useEffect, useRef, useState } from "react";
import Graph from "graphology";
import { Sigma } from "sigma";
import { useParams, useLocation } from "react-router-dom";

/**
 * WalletGraph
 *
 * Usage:
 *  - route example: /graph/:address
 *  - or query param: /graph?address=1FyCD...
 *
 * Expects REACT_APP_BLOCKCYPHER_TOKEN in env (create-react-app / Vite style).
 */

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

export default function WalletGraph() {
  const { address: paramAddress } = useParams() || {};
  const query = useQuery();
  const queryAddress = query.get("address");
  const address = (paramAddress || queryAddress || "1FyCD8kp9ekiTTgdyhFtZRgzR1QCHV4i84").trim();

  const containerRef = useRef(null);
  const sigmaRef = useRef(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (sigmaRef.current) {
        try {
          sigmaRef.current.kill();
        } catch (e) {
          // ignore
        }
        sigmaRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!address) {
      setError("No address provided in URL (route or ?address=).");
      setLoading(false);
      return;
    }

    setError(null);
    setLoading(true);
    setSummary(null);

    // read token from env (CRA/Vite)
    const token = "51eaeb12a21b4a4f85082d5b7c86ec44" || "";

    const url = `https://api.blockcypher.com/v1/btc/main/addrs/${encodeURIComponent(
      address
    )}/full?limit=50${token ? `&token=${encodeURIComponent(token)}` : ""}`;

    let mounted = true;

    (async () => {
      try {
        // fetch with a single retry if 429/5xx
        let res = await fetch(url);
        if (res.status >= 500 || res.status === 429) {
          // short wait and retry once
          await new Promise((r) => setTimeout(r, 800));
          res = await fetch(url);
        }

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`BlockCypher error: ${res.status} ${text}`);
        }

        const j = await res.json();
        if (!mounted) return;

        const txs = j.txs || [];
        const centerId = address;

        // build an empty graph and add center node
        const g = new Graph();
        g.addNode(centerId, {
          label: centerId,
          x: 0,
          y: 0,
          size: 20,
          color: "#2ECC71",
        });

        // accumulate counterparties map
        // cp -> { in: number (cp->wallet), out: number (wallet->cp), txs: Set }
        const cpMap = {};

        txs.forEach((tx) => {
          const inputs = tx.inputs || [];
          const outputs = tx.outputs || [];

          const inAddrs = new Set();
          inputs.forEach((vin) => (vin.addresses || []).forEach((a) => inAddrs.add(a)));

          const outAddrs = new Set();
          outputs.forEach((vout) => (vout.addresses || []).forEach((a) => outAddrs.add(a)));

          const walletInInputs = inAddrs.has(centerId);
          const walletInOutputs = outAddrs.has(centerId);

          // If wallet appears in outputs => money flowed into wallet from inputs (cp -> wallet)
          if (walletInOutputs) {
            inputs.forEach((vin) =>
              (vin.addresses || []).forEach((a) => {
                if (a === centerId) return;
                const rec = (cpMap[a] = cpMap[a] || { in: 0, out: 0, txs: new Set() });
                rec.in += 1;
                rec.txs.add(tx.hash);
              })
            );
          }

          // If wallet appears in inputs => money flowed out to outputs (wallet -> cp)
          if (walletInInputs) {
            outputs.forEach((vout) =>
              (vout.addresses || []).forEach((a) => {
                if (a === centerId) return;
                const rec = (cpMap[a] = cpMap[a] || { in: 0, out: 0, txs: new Set() });
                rec.out += 1;
                rec.txs.add(tx.hash);
              })
            );
          }
        });

        const cps = Object.keys(cpMap);
        const n = Math.max(cps.length, 1);
        const radius = Math.max(4, 3 + n / 6); // radius scales with number of nodes

        // Add CP nodes and appropriate edges (use valid Sigma edge types)
        cps.forEach((cp, i) => {
          const meta = cpMap[cp];
          const angle = (2 * Math.PI * i) / n;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;

          g.addNode(cp, {
            label: cp,
            x,
            y,
            size: 6 + Math.min(12, meta.txs.size),
            color: "#F39C12",
          });

          const hasIn = meta.in > 0; // cp -> wallet
          const hasOut = meta.out > 0; // wallet -> cp

          // We'll add directed edges using valid sigma edge types:
          // - "arrow" (simple arrow)
          // - "curvedArrow" (curved arrow to avoid overlap when both directions exist)
          if (hasIn && !hasOut) {
            // cp -> wallet
            const edgeId = `e-${cp}-to-${centerId}`;
            if (!g.hasEdge(edgeId)) {
              g.addEdgeWithKey(edgeId, cp, centerId, {
                label: `${meta.txs.size} tx(s)`,
                color: "#E74C3C",
                type: "arrow",
                size: Math.min(6, 1 + meta.txs.size),
              });
            }
          } else if (hasOut && !hasIn) {
            // wallet -> cp
            const edgeId = `e-${centerId}-to-${cp}`;
            if (!g.hasEdge(edgeId)) {
              g.addEdgeWithKey(edgeId, centerId, cp, {
                label: `${meta.txs.size} tx(s)`,
                color: "#3498DB",
                type: "arrow",
                size: Math.min(6, 1 + meta.txs.size),
              });
            }
          } else if (hasIn && hasOut) {
            // both directions — create two edges (curved arrows to avoid overlap)
            const e1 = `e-${cp}-to-${centerId}-in`;
            const e2 = `e-${centerId}-to-${cp}-out`;
            if (!g.hasEdge(e1)) {
              g.addEdgeWithKey(e1, cp, centerId, {
                label: `${meta.txs.size} tx(s)`,
                color: "#9B59B6",
                type: "curvedArrow",
                size: Math.min(6, 1 + meta.txs.size),
              });
            }
            if (!g.hasEdge(e2)) {
              g.addEdgeWithKey(e2, centerId, cp, {
                label: `${meta.txs.size} tx(s)`,
                color: "#9B59B6",
                type: "curvedArrow",
                size: Math.min(6, 1 + meta.txs.size),
              });
            }
          }
        });

        // If no counterparties found in the fetched txs, add a small placeholder node
        if (cps.length === 0) {
          const placeholderId = `${centerId}-empty`;
          g.addNode(placeholderId, {
            label: "(no counterparties in fetched txs)",
            x: 2,
            y: 2,
            size: 6,
            color: "#95A5A6",
          });
          const edgeId = `e-${centerId}-placeholder`;
          if (!g.hasEdge(edgeId)) {
            g.addEdge(edgeId, centerId, placeholderId, {
              color: "#95A5A6",
              type: "line",
            });
          }
        }

        // kill previous sigma if any
        if (sigmaRef.current) {
          try {
            sigmaRef.current.kill();
          } catch (e) {
            // ignore
          }
          sigmaRef.current = null;
        }

        // mount Sigma
        sigmaRef.current = new Sigma(g, containerRef.current, {
          renderLabels: true,
          labelThreshold: 6,
        });

        // small summary for UI
        setSummary({
          address: j.address || address,
          n_tx: j.n_tx ?? txs.length,
          total_received: j.total_received ?? j.total_received,
          total_sent: j.total_sent ?? j.total_sent,
          counterparties: cps.length,
        });

        setLoading(false);
      } catch (err) {
        if (!mounted) return;
        console.error("WalletGraph fetch/parse error:", err);
        setError(String(err?.message || err));
        setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [address]);

  return (
    <div className="p-4 ">
      <h2 className="text-xl font-semibold mb-2">Wallet Graph</h2>

      {!address ? (
        <div className="text-red-600">No address provided. Put address in URL param or ?address=</div>
      ) : (
        <>
          <div className="mb-2">
            <strong>Address:</strong> {address}
            {summary ? (
              <span className="ml-4 text-sm text-gray-600">
                txs: {summary.n_tx} • counterparties: {summary.counterparties}
              </span>
            ) : null}
          </div>

          {loading && <div className="text-gray-600 mb-2">Loading transactions & building graph…</div>}
          {error && <div className="text-red-600 mb-2">Error: {error}</div>}

          <div
            ref={containerRef}
            style={{
              width: "100%",
              height: "600px",
              borderRadius: 10,
              border: "1px solid #e6e6e6",
              background: "#cbd4fd",
            }}
          />

          <div className="mt-3 text-sm text-gray-500">
            Edge color:
            <span style={{ color: "#E74C3C", marginLeft: 8 }}> &nbsp; input ⟶ wallet</span>{" "}
            <span style={{ color: "#3498DB", marginLeft: 8 }}> &nbsp; wallet ⟶ output</span>{" "}
            <span style={{ color: "#9B59B6", marginLeft: 8 }}> &nbsp; both</span>
          </div>
        </>
      )}
    </div>
  );
}
