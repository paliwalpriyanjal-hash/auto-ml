import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { 
  Sparkles, 
  Cpu, 
  GitCompare, 
  Wand2, 
  Layers, 
  Clock, 
  PlayCircle,
  Menu,
  X,
  Activity,
  ShieldCheck,
  Sliders,
  Filter,
  FileText,
  ActivitySquare,
  Shield
} from "lucide-react";
import { getApiBase, getAuthToken } from "../utils/api";

const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState(null);
  const location = useLocation();

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${getApiBase()}/test`, { method: "GET" });
        if (isMounted) setBackendOnline(res.ok);
      } catch {
        if (isMounted) setBackendOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 20000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navLinks = [
    { name: "Overview", path: "/overview", icon: Activity },
    { name: "Model Training", path: "/automl", icon: Cpu },
    { name: "Compare Models", path: "/compare", icon: GitCompare },
    { name: "Data Cleaning", path: "/clean", icon: Wand2 },
    { name: "Saved Models", path: "/models", icon: Layers },
    { name: "Predictions", path: "/predict", icon: PlayCircle },
    { name: "Explainability", path: "/explain", icon: ShieldCheck },
    { name: "Tuning", path: "/tune", icon: Sliders },
    { name: "Features", path: "/features", icon: Filter },
    { name: "Reports", path: "/reports", icon: FileText },
    { name: "Monitoring", path: "/monitoring", icon: ActivitySquare },
    { name: "Experiments", path: "/experiments", icon: Clock },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-[#070711]/85 backdrop-blur-xl border-b border-white/[0.08] transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        
        {/* Brand Logo */}
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#8265FF] to-[#A855F7] flex items-center justify-center text-white shadow-lg shadow-[#8265FF]/25 group-hover:shadow-[#8265FF]/40 transition-all duration-300">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div className="flex flex-col">
            <span className="font-extrabold text-xl tracking-tight text-white flex items-center gap-1.5">
              AutoVeyra
              <span className="text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 rounded-full bg-[#8265FF]/20 text-[#A855F7] border border-[#8265FF]/30">
                PRO
              </span>
            </span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <div className="hidden lg:flex items-center gap-1 bg-[#0D0D1B]/80 p-1 rounded-xl border border-white/[0.06]">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = isActive(link.path);
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                  active
                    ? "bg-[#171630] text-white shadow-sm border border-[#8265FF]/30 shadow-[#8265FF]/10"
                    : "text-[#A4A4B8] hover:text-white hover:bg-white/[0.04]"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${active ? "text-[#8265FF]" : "text-[#A4A4B8]"}`} />
                {link.name}
              </Link>
            );
          })}
        </div>

        {/* Status Indicator & Quick CTA */}
        <div className="hidden sm:flex items-center gap-4">
          <div 
            className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#0D0D1B] border border-white/[0.08] text-xs text-[#A4A4B8]"
            title={`FastAPI Backend: ${backendOnline ? 'Online (Port 8000)' : 'Checking / Offline'}`}
          >
            <span className={`w-2 h-2 rounded-full animate-pulse ${
              backendOnline === true ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" :
              backendOnline === false ? "bg-rose-400 shadow-[0_0_8px_#f87171]" : "bg-amber-400"
            }`} />
            <span className="text-[11px] font-medium tracking-wide">
              {backendOnline === true ? "API Live" : backendOnline === false ? "API Offline" : "Connecting"}
            </span>
          </div>

          <Link
            to="/login"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-[#0D0D1B] border border-white/[0.08] hover:border-[#8265FF]/40 text-xs text-[#A4A4B8] hover:text-white transition-all"
            title="Authentication & Session Security"
          >
            <Shield className="w-3.5 h-3.5 text-[#8265FF]" />
            <span className="text-[11px] font-medium">{getAuthToken() ? "Account" : "Sign In"}</span>
          </Link>

          <Link
            to="/automl"
            className="px-4 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-md shadow-[#8265FF]/20 hover:shadow-[#8265FF]/40 transition-all duration-200 flex items-center gap-1.5"
          >
            <span>Start Training</span>
          </Link>
        </div>

        {/* Mobile menu button */}
        <div className="flex lg:hidden items-center gap-2">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg bg-[#111126] border border-white/[0.08] text-gray-300 hover:text-white"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-b border-white/[0.08] bg-[#070711]/95 backdrop-blur-2xl px-4 pt-2 pb-6 space-y-2">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = isActive(link.path);
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium ${
                  active
                    ? "bg-[#171630] text-white border border-[#8265FF]/40"
                    : "text-[#A4A4B8] hover:text-white hover:bg-white/[0.04]"
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? "text-[#8265FF]" : "text-[#A4A4B8]"}`} />
                {link.name}
              </Link>
            );
          })}
          <div className="pt-4 border-t border-white/[0.08] flex items-center justify-between">
            <span className="text-xs text-[#A4A4B8]">System: {backendOnline ? "Online" : "Disconnected"}</span>
            <Link
              to="/automl"
              onClick={() => setMobileMenuOpen(false)}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] text-white"
            >
              Start Building
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
