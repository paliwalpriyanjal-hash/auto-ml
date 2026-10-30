// client/src/components/Navbar.jsx
import React, { useState, useEffect, useRef } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Sparkles,
  Compass,
  Activity,
  BarChart3,
  Wand2,
  Filter,
  Cpu,
  GitCompare,
  Layers,
  PlayCircle,
  ShieldCheck,
  Sliders,
  Clock,
  ActivitySquare,
  FileText,
  Shield,
  ChevronDown,
  Menu,
  X
} from "lucide-react";
import { getApiBase, getAuthToken } from "../utils/api";

const NAV_GROUPS = [
  {
    id: "workspace",
    name: "Workspace",
    items: [
      { name: "Overview", path: "/overview", icon: Activity, desc: "Platform metrics and project history" },
      { name: "Guided Workspace", path: "/workspace", icon: Compass, desc: "10-step guided machine learning studio", badge: "New" },
    ],
  },
  {
    id: "data",
    name: "Data Studio",
    items: [
      { name: "Adaptive EDA", path: "/eda", icon: BarChart3, desc: "Automated statistical analysis & distributions" },
      { name: "Data Cleaning", path: "/clean", icon: Wand2, desc: "Imputation, deduplication & outliers" },
      { name: "Feature Engineering", path: "/features", icon: Filter, desc: "Transformations & feature selection" },
    ],
  },
  {
    id: "models",
    name: "Machine Learning",
    items: [
      { name: "AutoML Training", path: "/automl", icon: Cpu, desc: "Train multiple ML algorithms" },
      { name: "Compare Models", path: "/compare", icon: GitCompare, desc: "Cross-validation leaderboards & benchmarks" },
      { name: "Saved Models", path: "/models", icon: Layers, desc: "Model registry and persistent artifacts" },
      { name: "Predictions", path: "/predict", icon: PlayCircle, desc: "Real-time model inference" },
      { name: "Explainability", path: "/explain", icon: ShieldCheck, desc: "SHAP and permutation importances" },
    ],
  },
  {
    id: "experiments",
    name: "Experiments",
    items: [
      { name: "Workbench", path: "/experiments", icon: Clock, desc: "Experiment tracking & lineage DAG" },
      { name: "Hyperparameter Tuning", path: "/tune", icon: Sliders, desc: "Optuna Bayesian hyperparameter search" },
    ],
  },
  {
    id: "governance",
    name: "Intelligence",
    items: [
      { name: "Model Monitoring", path: "/monitoring", icon: ActivitySquare, desc: "Drift detection & serving telemetry" },
      { name: "Report Studio", path: "/reports", icon: FileText, desc: "Unified project documentation & ZIP exports" },
    ],
  },
];

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null);
  const [showDemoBanner, setShowDemoBanner] = useState(() => {
    try {
      return sessionStorage.getItem("autoveyra_demo_banner_dismissed") !== "true";
    } catch {
      return true;
    }
  });
  const location = useLocation();
  const navRef = useRef(null);

  const dismissDemoBanner = () => {
    setShowDemoBanner(false);
    try {
      sessionStorage.setItem("autoveyra_demo_banner_dismissed", "true");
    } catch (err) {
      console.debug("Failed to store demo banner state in sessionStorage", err);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        let res = await fetch(`${getApiBase()}/health`, { method: "GET" }).catch(() => null);
        if (!res || !res.ok) {
          res = await fetch(`${getApiBase()}/test`, { method: "GET" }).catch(() => null);
        }
        if (isMounted) setBackendOnline(Boolean(res && res.ok));
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

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (navRef.current && !navRef.current.contains(event.target)) {
        setActiveDropdown(null);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Close menus on route change
  useEffect(() => {
    setActiveDropdown(null);
    setMobileMenuOpen(false);
  }, [location.pathname]);

  const isGroupActive = (group) =>
    group.items.some((item) => item.path === location.pathname);

  return (
    <nav
      ref={navRef}
      className="fixed top-0 left-0 right-0 z-50 bg-[#070711]/90 backdrop-blur-xl border-b border-white/[0.08] transition-all"
    >
      {/* Ephemeral Demo Cloud Banner */}
      {showDemoBanner && (
        <div className="bg-gradient-to-r from-[#8265FF]/20 via-[#4E75FF]/20 to-[#8265FF]/20 border-b border-[#8265FF]/30 px-3 py-1.5 text-xs text-[#C5C8D8] flex items-center justify-between gap-2">
          <div className="max-w-7xl mx-auto flex items-center gap-2 flex-1 justify-center text-center">
            <span className="font-semibold text-[#8265FF] inline-flex items-center gap-1">
              ⚡ Demo Environment:
            </span>
            <span className="hidden sm:inline">
              Running on free cloud tier. Backend hibernates after inactivity (~50s cold start). Filesystem storage is ephemeral; models and datasets reset upon redeploy.
            </span>
            <span className="sm:hidden">
              Free cloud demo (ephemeral storage; cold starts take ~50s).
            </span>
          </div>
          <button
            onClick={dismissDemoBanner}
            className="text-white/60 hover:text-white px-1.5 py-0.5 text-xs rounded hover:bg-white/10 transition-colors"
            title="Dismiss notice"
            aria-label="Dismiss notice"
          >
            ✕
          </button>
        </div>
      )}

      <div className="max-w-7xl mx-auto flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        
        {/* Brand Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#8265FF] to-[#A855F7] flex items-center justify-center text-white shadow-md shadow-[#8265FF]/20 group-hover:shadow-[#8265FF]/35 transition-all">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="font-extrabold text-lg tracking-tight text-white flex items-center gap-1.5">
            AutoVeyra
            <span className="text-[9px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded-full bg-[#8265FF]/15 text-[#C084FC] border border-[#8265FF]/25">
              PRO
            </span>
          </span>
        </Link>

        {/* Desktop Grouped Navigation */}
        <div className="hidden lg:flex items-center gap-1 bg-[#0F0F22]/90 p-1 rounded-xl border border-white/[0.08]">
          {NAV_GROUPS.map((group) => {
            const active = isGroupActive(group);
            const isOpen = activeDropdown === group.id;

            return (
              <div key={group.id} className="relative">
                <button
                  type="button"
                  onClick={() => setActiveDropdown(isOpen ? null : group.id)}
                  onMouseEnter={() => setActiveDropdown(group.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                    active || isOpen
                      ? "bg-[#16152F] text-white shadow-sm border border-[#8265FF]/35 text-[#8265FF]"
                      : "text-[#9FA2B6] hover:text-white hover:bg-white/[0.04]"
                  }`}
                  aria-expanded={isOpen}
                >
                  <span>{group.name}</span>
                  <ChevronDown
                    className={`w-3.5 h-3.5 transition-transform duration-200 ${
                      isOpen ? "rotate-180 text-[#8265FF]" : "opacity-60"
                    }`}
                  />
                </button>

                {/* Dropdown Menu */}
                {isOpen && (
                  <div
                    onMouseLeave={() => setActiveDropdown(null)}
                    className="absolute top-full left-0 mt-2 w-72 p-2 rounded-2xl bg-[#0F0F22] border border-white/[0.12] shadow-2xl shadow-black/60 backdrop-blur-2xl animate-in fade-in zoom-in-95 duration-150 z-50 space-y-1"
                  >
                    {group.items.map((item) => {
                      const Icon = item.icon;
                      const isItemActive = location.pathname === item.path;
                      return (
                        <Link
                          key={item.path}
                          to={item.path}
                          onClick={() => setActiveDropdown(null)}
                          className={`flex items-start gap-3 p-2.5 rounded-xl transition-all ${
                            isItemActive
                              ? "bg-[#16152F] text-white border border-[#8265FF]/30"
                              : "text-[#9FA2B6] hover:text-white hover:bg-white/[0.04]"
                          }`}
                        >
                          <div
                            className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                              isItemActive
                                ? "bg-[#8265FF] text-white"
                                : "bg-white/[0.06] text-[#9FA2B6]"
                            }`}
                          >
                            <Icon className="w-3.5 h-3.5" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span className="font-semibold text-xs text-white">
                                {item.name}
                              </span>
                              {item.badge && (
                                <span className="text-[9px] uppercase font-bold px-1.5 py-0.2 rounded-full bg-[#8265FF]/20 text-[#8265FF]">
                                  {item.badge}
                                </span>
                              )}
                            </div>
                            <p className="text-[11px] text-[#64677E] truncate">
                              {item.desc}
                            </p>
                          </div>
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Right Action Controls: API status, Sign In, Guided Studio CTA */}
        <div className="hidden sm:flex items-center gap-3">
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#0F0F22] border border-white/[0.08] text-[11px] text-[#9FA2B6]"
            title={`FastAPI Backend: ${backendOnline ? "Online (Port 8000)" : "Connecting / Offline"}`}
          >
            <span
              className={`w-2 h-2 rounded-full animate-pulse ${
                backendOnline === true
                  ? "bg-emerald-400 shadow-[0_0_8px_#34d399]"
                  : backendOnline === false
                  ? "bg-rose-400 shadow-[0_0_8px_#f87171]"
                  : "bg-amber-400"
              }`}
            />
            <span className="font-medium">
              {backendOnline === true ? "API Live" : backendOnline === false ? "Offline" : "Connecting"}
            </span>
          </div>

          <Link
            to="/login"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#0F0F22] border border-white/[0.08] hover:border-[#8265FF]/40 text-xs text-[#9FA2B6] hover:text-white transition-all font-medium"
          >
            <Shield className="w-3.5 h-3.5 text-[#8265FF]" />
            <span>{getAuthToken() ? "Account" : "Sign In"}</span>
          </Link>

          <Link
            to="/workspace"
            className="px-4 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-md shadow-[#8265FF]/20 hover:shadow-[#8265FF]/35 transition-all flex items-center gap-1.5"
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Launch Studio</span>
          </Link>
        </div>

        {/* Mobile menu toggle */}
        <div className="flex lg:hidden items-center gap-2">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg bg-[#0F0F22] border border-white/[0.08] text-[#9FA2B6] hover:text-white"
            aria-label="Toggle navigation"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-b border-white/[0.08] bg-[#070711]/98 backdrop-blur-2xl px-4 pt-3 pb-6 space-y-4 max-h-[80vh] overflow-y-auto">
          {NAV_GROUPS.map((group) => (
            <div key={group.id} className="space-y-1">
              <span className="text-[10px] uppercase font-bold tracking-wider text-[#64677E] px-2 block">
                {group.name}
              </span>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const active = location.pathname === item.path;
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-medium transition-all ${
                        active
                          ? "bg-[#16152F] text-white border border-[#8265FF]/40"
                          : "text-[#9FA2B6] hover:text-white hover:bg-white/[0.04]"
                      }`}
                    >
                      <Icon className={`w-4 h-4 ${active ? "text-[#8265FF]" : "text-[#64677E]"}`} />
                      <span>{item.name}</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}

          <div className="pt-3 border-t border-white/[0.08] flex items-center justify-between">
            <span className="text-xs text-[#9FA2B6]">
              Backend: {backendOnline ? "Live (8000)" : "Disconnected"}
            </span>
            <Link
              to="/workspace"
              onClick={() => setMobileMenuOpen(false)}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-[#8265FF] to-[#A855F7] text-white"
            >
              Launch Studio
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}
