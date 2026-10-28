import React from "react";
import { Link } from "react-router-dom";
import { Sparkles, Github, Terminal, ShieldCheck, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="bg-[#05050C] border-t border-white/[0.06] text-[#A4A4B8] text-sm relative z-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
          
          {/* Brand Column */}
          <div className="md:col-span-2 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#8265FF] to-[#A855F7] flex items-center justify-center text-white shadow-md shadow-[#8265FF]/20">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <span className="font-extrabold text-lg tracking-tight text-white">
                AutoVeyra
              </span>
            </div>
            <p className="text-xs text-[#A4A4B8] max-w-sm leading-relaxed">
              Industrial-grade Automated Machine Learning platform engineered for rapid data exploration, multi-algorithm benchmarking, reproducible pipelines, and instant inference.
            </p>
            <div className="flex items-center gap-3 text-xs text-[#A4A4B8]">
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> Production-Ready Architecture
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Terminal className="w-3.5 h-3.5 text-[#8265FF]" /> FastAPI Powered
              </span>
            </div>
          </div>

          {/* Quick Workspaces */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-white mb-3">Workspaces</h4>
            <ul className="space-y-2 text-xs">
              <li><Link to="/overview" className="hover:text-white transition-colors">Overview Dashboard</Link></li>
              <li><Link to="/automl" className="hover:text-white transition-colors">Model Training</Link></li>
              <li><Link to="/compare" className="hover:text-white transition-colors">Compare Algorithms</Link></li>
              <li><Link to="/clean" className="hover:text-white transition-colors">Advanced Cleaning</Link></li>
            </ul>
          </div>

          {/* Registry & Deployment */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-white mb-3">Deployment & Audit</h4>
            <ul className="space-y-2 text-xs">
              <li><Link to="/models" className="hover:text-white transition-colors">Model Registry</Link></li>
              <li><Link to="/predict" className="hover:text-white transition-colors">Inference Testing</Link></li>
              <li><Link to="/experiments" className="hover:text-white transition-colors">Experiment History</Link></li>
            </ul>
          </div>

        </div>

        {/* Bottom Bar */}
        <div className="pt-6 border-t border-white/[0.04] flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
          <p>© {new Date().getFullYear()} AutoVeyra AI Systems. All rights reserved.</p>
          <div className="flex items-center gap-4 text-[#A4A4B8]">
            <span>FastAPI 0.162+</span>
            <span>•</span>
            <span>scikit-learn 1.6+</span>
            <span>•</span>
            <span>React 19</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
