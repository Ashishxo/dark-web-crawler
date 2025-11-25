import React, { useEffect, useRef } from "react";
import Graph from "graphology";
import { Sigma } from "sigma";

const GraphComponent = () => {
  const containerRef = useRef(null);
  const sigmaInstance = useRef(null);

  const addressData = {
    address: "1MainWallet",
    transactions: [
      { txid: "abcd1234efgh5678", status: "input" },
      { txid: "ijkl9012mnop3456", status: "output" },
      { txid: "qrst7890uvwx1234", status: "input" },
      { txid: "wxyz9876lmno5432", status: "output" },
      { txid: "tx0001alpha", status: "input" },
      { txid: "tx0002beta", status: "output" },
      { txid: "tx0003gamma", status: "input" },
    ],
  };

  useEffect(() => {
    const graph = new Graph();
    const walletAddress = addressData.address;

    graph.addNode(walletAddress, {
      label: walletAddress,
      x: 0,
      y: 0,
      size: 20,
      color: "#2ECC71",
    });

    addressData.transactions.forEach((tx, idx) => {
      const txNodeId = `tx-${tx.txid}-${idx}`;
      const angle = (2 * Math.PI * idx) / addressData.transactions.length;
      const radius = 4;

      graph.addNode(txNodeId, {
        label: tx.txid.slice(0, 10),
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        size: 10,
        color: tx.status === "input" ? "#E74C3C" : "#3498DB",
      });

      graph.addEdge(walletAddress, txNodeId);
    });

    if (sigmaInstance.current) sigmaInstance.current.kill();

    sigmaInstance.current = new Sigma(graph, containerRef.current, {
      renderLabels: true,
    });

    return () => {
      sigmaInstance.current?.kill();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "600px",
        borderRadius: "10px",
        marginTop: "20px",
      }}
    />
  );
};

export default GraphComponent;
