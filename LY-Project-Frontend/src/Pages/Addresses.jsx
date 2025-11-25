// src/Pages/Addresses.jsx
import React, { useEffect, useState } from "react";

function formatDateISO(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? String(iso) : d.toLocaleString();
}

function truncate(text, n = 200) {
  if (!text) return "—";
  return text.length > n ? text.slice(0, n) + "…" : text;
}

export default function Addresses() {
  const [addresses, setAddresses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAddresses = async () => {
      try {
        const res = await fetch("http://localhost:8000/addresses?limit=200");
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt || "Failed to fetch addresses");
        }
        const data = await res.json();
        console.log("raw /addresses response:", data);

        // Ensure stable ordering (API returned newest first in your example)
        const normalized = (data || []).map((it) => ({
          id: it.id,
          address: it.address,
          context_snippet: it.context_snippet,
          url: it.url,
          title: it.title,
          saved_at: it.saved_at,
          depth: it.depth,
          session_id: it.session_id,
        }));

        setAddresses(normalized);
      } catch (err) {
        console.error("Could not fetch extracted addresses:", err);
        setError(err.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchAddresses();
  }, []);

  if (loading) return <div className="p-10 text-lg text-gray-600">Loading extracted addresses...</div>;
  if (error) return <div className="p-10 text-lg text-red-600">Error: {error}</div>;
  if (!addresses.length) return <div className="p-10 text-lg text-gray-600">No addresses found yet.</div>;

  return (
    <div className="p-10 flex flex-col items-center">
      <h1 className="text-2xl font-bold text-[#4A5AB5] mb-6">Extracted Bitcoin Addresses</h1>

      <div className="overflow-auto shadow-md rounded-lg w-11/12">
        <table className="min-w-full bg-white">
          <thead className="bg-[#6756FF] text-white">
            <tr>
              <th className="py-3 px-4 text-left w-12">#</th>
              <th className="py-3 px-4 text-left">Address</th>
              <th className="py-3 px-4 text-left">Context (snippet)</th>
              <th className="py-3 px-4 text-left">Source</th>
              <th className="py-3 px-4 text-left">Timestamp</th>
            </tr>
          </thead>

          <tbody>
            {addresses.map((it, idx) => (
              <tr key={it.id ?? idx} className="border-b hover:bg-gray-50 align-top">
                <td className="py-3 px-4 align-top">{idx + 1}</td>

                <td className="py-3 px-4 font-mono text-sm break-words max-w-[28rem]">{it.address}</td>

                <td className="py-3 px-4 text-sm">
                  <div title={it.context_snippet}>{truncate(it.context_snippet, 220)}</div>
                </td>

                <td className="py-3 px-4">
                  {it.url ? (
                    <a href={it.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline break-words">
                      {it.url}
                    </a>
                  ) : (
                    it.title ?? "—"
                  )}
                  {/* small meta row */}
                  <div className="text-xs text-gray-500 mt-1">
                    {it.title ? <span>{it.title} • </span> : null}
                    {typeof it.depth !== "undefined" ? <span>depth: {it.depth}</span> : null}
                  </div>
                </td>

                <td className="py-3 px-4">{formatDateISO(it.saved_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
