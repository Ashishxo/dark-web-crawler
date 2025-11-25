// src/App.jsx
import { useState } from "react";
import DarkWebSpider from "./Pages/DarkWebSpider";
import { Link, Outlet } from "react-router-dom";
import Addresses from "./Pages/Addresses";
import { Routes, Route } from "react-router-dom";

function App() {
  
  return (
    <>
    <div className="w-screen h-[32px] bg-[#6756FF]"></div>
    <div className="main w-full flex flex-col h-screen">
      
      <div className="bg-[#D8DFFF] border border-white/40 rounded-full pt-3 pb-2 px-10 mx-auto mt-8 mb-12 flex gap-8 items-center justify-center">
        <Link to="/" className="text-lg text-[#4A5AB5] font-medium mb-2 hover:bg-[#c1ccfd] px-4 py-1 rounded-3xl cursor-pointer">Dark Web Spider</Link>
        <Link to="/addresses" className="text-lg text-[#4A5AB5] font-medium mb-2 hover:bg-[#c1ccfd] px-4 py-1 rounded-3xl cursor-pointer">Extracted Addresses</Link>
        <Link to="/graph" className="text-lg text-[#4A5AB5] font-medium mb-2 hover:bg-[#c1ccfd] px-4 py-1 rounded-3xl cursor-pointer">Graph Visualizer</Link>
        <div className="text-lg text-[#4A5AB5] font-medium mb-2 hover:bg-[#c1ccfd] px-4 py-1 rounded-3xl cursor-pointer">Malicious Wallets</div>
      </div>

      <Outlet/>
    

      
    </div>
    </>
  );
}

export default App;
