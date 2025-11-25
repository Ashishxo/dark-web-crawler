// src/CrawlPage.jsx
import { useEffect, useState, useRef } from "react";
import { useLocation } from "react-router-dom";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

export default function CrawlPage() {
  const query = useQuery();
  const session = query.get("session");
  const [lines, setLines] = useState([]);
  const [stopped, setStopped] = useState(false);
  const evtRef = useRef(null);

  useEffect(() => {
    if (!session) return;
    const es = new EventSource(`http://localhost:8000/stream/${session}`);
    evtRef.current = es;

    es.onmessage = (e) => {
      if (!e.data) return;
      setLines((prev) => [...prev, e.data]);
      if (e.data === "==DONE==" || e.data === "🛑 Crawl stopped by user.") {
        // close after final message
        try { es.close(); } catch {}
        setStopped(true);
      }
    };

    es.onerror = (err) => {
      console.error("SSE error", err);
      try { es.close(); } catch {}
    };

    return () => {
      try { es.close(); } catch {}
      evtRef.current = null;
    };
  }, [session]);

  const stopCrawl = async () => {
    if (!session) return;
    try {
      await fetch(`http://localhost:8000/stop/${session}`, { method: "POST" });
      if (evtRef.current) evtRef.current.close();
      setStopped(true);
    } catch (err) {
      console.error(err);
      alert("Failed to request stop");
    }
  };

  return (
    <div className="h-screen">
      <div className="w-screen h-6 bg-[#6756FF]"></div>
      <div style={{ padding: 20 }}>
        
        <h2>Live Crawl — Session {session}</h2>

        <div style={{
          whiteSpace: "pre-wrap",
          background: "#111",
          color: "#e6e6e6",
          padding: 12,
          height: "75vh",
          overflow: "auto",
          borderRadius: 8,
          marginBottom: "16px",
          marginTop: "10px"
        }}>
          {lines.map((l, i) => <div key={i}>{l}</div>)}
        </div>

        <div style={{ marginBottom: 12 }}>
          <button
            onClick={stopCrawl}
            disabled={stopped}
            style={{
              background: stopped ? "gray" : "#6756FF",
              color: "white",
              padding: "8px 16px",
              borderRadius: "8px",
              cursor: stopped ? "not-allowed" : "pointer",
            }}
          >
            {stopped ? "Stopped" : "Stop Crawl"}
          </button>
        </div>

      </div>
    </div>
  );
}
