import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

function formatDateISO(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? String(iso) : d.toLocaleString();
}

function truncate(text, n = 200) {
  if (!text) return "—";
  return text.length > n ? text.slice(0, n) + "…" : text;
}

export default function MaliciousWallet() {
  const [addresses, setAddresses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const fetchAddresses = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/addresses?limit=500");
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Failed to fetch addresses");
      }
      const data = await res.json();

      // keep only malicious (is_illicit === 1)
      const malicious = (data || []).filter((it) => Number(it.is_illicit) === 1);

      // normalize fields (newest first)
      const normalized = malicious.map((it) => ({
        id: it.id,
        address: it.address,
        context_snippet: it.context_snippet,
        url: it.url,
        title: it.title,
        saved_at: it.saved_at,
        depth: it.depth,
        session_id: it.session_id,
        nlp_label: it.nlp_label,   // <-- NEW FIELD
      }));

      setAddresses(normalized);
    } catch (err) {
      console.error("Could not fetch extracted addresses:", err);
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAddresses();
  }, []);

  const downloadCSV = () => {
    if (!addresses.length) return;
    const header = ["id", "address", "context_snippet", "url", "title", "saved_at", "depth", "session_id", "nlp_label"];
    const rows = addresses.map((r) =>
      header.map((h) => {
        const v = r[h];
        if (v === null || typeof v === "undefined") return "";
        return String(v).replace(/"/g, '""');
      })
    );
    const csv = [header.join(","), ...rows.map((r) => `"${r.join('","')}"`)].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `malicious_addresses_${new Date().toISOString().slice(0,19)}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  if (loading)
    return <div className="p-8 text-lg text-gray-600">Loading malicious addresses...</div>;

  if (error)
    return <div className="p-8 text-lg text-red-600">Error: {error}</div>;

  if (!addresses.length)
    return <div className="p-8 text-lg text-gray-600">No malicious addresses found yet.</div>;

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-[#4A5AB5]">Malicious / Illicit Wallets</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchAddresses}
            className="px-3 py-1 bg-[#6756FF] text-white rounded hover:brightness-95"
            title="Refresh list"
          >
            Refresh
          </button>
          <button
            onClick={downloadCSV}
            className="px-3 py-1 border rounded hover:bg-gray-50"
            title="Download CSV of shown addresses"
          >
            Export CSV
          </button>
        </div>
      </div>

      <div className="overflow-auto shadow rounded-lg">
        <table className="min-w-full bg-white table-fixed">
          <thead className="bg-[#6756FF] text-white">
            <tr>
              <th className="py-2 px-3 text-left w-10">#</th>
              <th className="py-2 px-3 text-left w-72">Address</th>
              <th className="py-2 px-3 text-left w-40">Type</th>
              <th className="py-2 px-3 text-left w-48">NLP Label</th> {/* NEW COLUMN */}
              <th className="py-2 px-3 text-left">Context (snippet)</th>
              <th className="py-2 px-3 text-left w-60">Source</th>
              <th className="py-2 px-3 text-left w-36">Timestamp</th>
            </tr>
          </thead>

          <tbody>
            {addresses.map((it, idx) => (
              <tr
                key={it.id ?? idx}
                onClick={() => navigate(`/graph/${it.address}`)}
                className="border-b hover:bg-gray-50 align-top"
              >
                <td className="py-2 px-3 align-top text-sm">{idx + 1}</td>

                <td className="py-2 px-3 font-mono text-sm break-words max-w-[28rem]">
                  <div className="break-words">{it.address}</div>
                </td>

                <td className="py-2 px-3">
                  <span className="inline-block bg-red-100 text-red-800 text-xs px-2 py-0.5 rounded">
                    Illicit
                  </span>
                </td>

                {/* NLP LABEL */}
                <td className="py-2 px-3 text-sm break-words">
                  <span className="inline-block bg-blue-100 text-blue-800 px-2 py-0.5 rounded text-xs">
                    {it.nlp_label || "—"}
                  </span>
                </td>

                <td className="py-2 px-3 text-sm">
                  <div title={it.context_snippet} className="whitespace-pre-wrap break-words">
                    {truncate(it.context_snippet, 220)}
                  </div>
                </td>

                <td className="py-2 px-3 text-sm break-words max-w-[24rem]">
                  {it.url ? (
                    <a
                      href={it.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 underline break-words"
                    >
                      {truncate(it.url, 80)}
                    </a>
                  ) : (
                    it.title ?? "—"
                  )}
                  <div className="text-xs text-gray-500 mt-1">
                    {it.title ? <span>{truncate(it.title, 60)} • </span> : null}
                    {typeof it.depth !== "undefined" ? <span>depth: {it.depth}</span> : null}
                  </div>
                </td>

                <td className="py-2 px-3 text-sm">{formatDateISO(it.saved_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
