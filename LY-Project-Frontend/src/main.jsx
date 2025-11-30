// src/main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import CrawlPage from "./Pages/CrawlPage.jsx";;
import "./index.css"; // if you have tailwind or other styles
import DarkWebSpider from "./Pages/DarkWebSpider.jsx";
import Addresses from "./Pages/Addresses.jsx";
import Graph from "./Pages/Graph.jsx";
import MaliciousWallet from "./Pages/MaliciousWallet.jsx";


ReactDOM.createRoot(document.getElementById("root")).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App />}>
        <Route index element={<DarkWebSpider />} /> 
        <Route path="/addresses" element={<Addresses />} />
        <Route path="/graph" element={<Graph />} />
        <Route path="/graph/:address" element={<Graph />} />
        <Route path="/malicious-wallets" element={<MaliciousWallet />} />
        
      </Route>
      <Route path="/crawl" element={<CrawlPage />} />
      
    </Routes>
  </BrowserRouter>
);
