import React, {useState} from 'react'
import GraphComponent from './Graph';

function DarkWebSpider() {


    const [startUrl, setStartUrl] = useState("");
    const [maxDepth, setMaxDepth] = useState(2);

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
        const res = await fetch("http://localhost:8000/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ start_url: startUrl, max_depth: maxDepth }),
        });
        if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || "Failed to start");
        }
        const data = await res.json();
        const sessionId = data.session_id;
        // Open new tab with Crawl viewer
        const url = `${window.location.origin}/crawl?session=${sessionId}`;
        window.open(url, "_blank");
        } catch (err) {
        console.error(err);
        alert("Could not start crawl. Check backend console and CORS.");
        }
    };


  return (
    <div className="w-full h-full flex mt-2">
        <div className="w-3/5 h-full px-20 py-15">
          <h1 className="text-[#180047] font-medium text-[3rem]">Dark web Spider for</h1>
          <h1 className="bg-gradient-to-r from-[#7369FF] to-[#5EA1FF] bg-clip-text text-transparent font-semibold text-[3.5rem]">Bitcoin Wallet Analysis</h1>

          <form className="flex flex-col w-3/4 gap-4 mt-10" onSubmit={handleSubmit}>
            <input
              value={startUrl}
              onChange={(e) => setStartUrl(e.target.value)}
              type="text"
              placeholder="Enter Starting Onion Link Or A Keyword"
              className="bg-white p-3 px-8 rounded-4xl outline-none"
              required
            />
            <input
              value={maxDepth}
              onChange={(e) => setMaxDepth(e.target.value)}
              type="number"
              min={1}
              max={3}
              placeholder="Select Max Depth (Maximum 3)"
              className="bg-white p-3 px-8 rounded-4xl outline-none"
              required
            />
            <input type="submit" value="Start Crawling" className="bg-[#6756FF] text-white rounded-4xl p-3 px-5 font-semibold cursor-pointer"/>
          </form>
        </div>
        <div className="w-2/5 bg-[#EBEFFF] h-full rounded-tl-4xl px-10">
          <GraphComponent/>
        </div>
      </div>
  )
}

export default DarkWebSpider