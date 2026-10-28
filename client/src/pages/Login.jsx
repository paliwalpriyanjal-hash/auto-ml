import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Shield, Lock, User, Mail, ArrowRight, CheckCircle2, AlertCircle } from "lucide-react";
import { getApiBase, setAuthToken, extractErrorMessage } from "../utils/api";

export default function Login() {
  const navigate = useNavigate();
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("developer");
  const [authStatus, setAuthStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    fetch(`${getApiBase()}/auth/status`)
      .then((r) => r.json())
      .then((data) => setAuthStatus(data))
      .catch(() => {});
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    const apiBase = getApiBase();
    const endpoint = isRegister ? `${apiBase}/auth/register` : `${apiBase}/auth/login`;
    const payload = isRegister
      ? { username, password, email: email || null, role }
      : { username, password };

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const msg = await extractErrorMessage(res, isRegister ? "Registration failed" : "Login failed");
        throw new Error(msg);
      }

      const data = await res.json();
      if (isRegister) {
        setSuccess("Registration successful! You can now log in.");
        setIsRegister(false);
      } else {
        setAuthToken(data.access_token);
        setSuccess(`Welcome back, ${data.username || username}!`);
        setTimeout(() => {
          navigate("/monitoring");
        }, 700);
      }
    } catch (err) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-4 flex items-center justify-center">
      <div className="w-full max-w-md bg-[#0D0D1B]/95 border border-white/[0.08] rounded-2xl shadow-2xl p-8 backdrop-blur-xl">
        <div className="flex flex-col items-center text-center mb-8">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-[#8265FF] to-[#A855F7] flex items-center justify-center text-white shadow-lg shadow-[#8265FF]/30 mb-4">
            <Shield className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            {isRegister ? "Create AutoVeyra Account" : "Sign In to AutoVeyra"}
          </h1>
          <p className="text-sm text-[#A4A4B8] mt-1.5">
            Enterprise JWT Authentication & Model Governance
          </p>
          {authStatus && (
            <div className="mt-3 inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-[#171630] border border-white/[0.06] text-[#A4A4B8]">
              <span className={`w-2 h-2 rounded-full ${authStatus.auth_enabled ? "bg-amber-400" : "bg-emerald-400"}`} />
              <span>Enforcement: {authStatus.auth_enabled ? "Strict JWT Mode" : "Permissive / Dev Mode"}</span>
            </div>
          )}
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start gap-3">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-400" />
            <div>{error}</div>
          </div>
        )}

        {success && (
          <div className="mb-6 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs flex items-start gap-3">
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-400" />
            <div>{success}</div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-[#A4A4B8] mb-1.5">Username</label>
            <div className="relative">
              <User className="w-4 h-4 absolute left-3.5 top-3.5 text-[#A4A4B8]" />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="w-full bg-[#111126] border border-white/[0.08] focus:border-[#8265FF] rounded-xl pl-10 pr-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none transition-all"
              />
            </div>
          </div>

          {isRegister && (
            <div>
              <label className="block text-xs font-semibold text-[#A4A4B8] mb-1.5">Email (Optional)</label>
              <div className="relative">
                <Mail className="w-4 h-4 absolute left-3.5 top-3.5 text-[#A4A4B8]" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full bg-[#111126] border border-white/[0.08] focus:border-[#8265FF] rounded-xl pl-10 pr-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none transition-all"
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold text-[#A4A4B8] mb-1.5">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 absolute left-3.5 top-3.5 text-[#A4A4B8]" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-[#111126] border border-white/[0.08] focus:border-[#8265FF] rounded-xl pl-10 pr-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none transition-all"
              />
            </div>
          </div>

          {isRegister && (
            <div>
              <label className="block text-xs font-semibold text-[#A4A4B8] mb-1.5">Assigned Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full bg-[#111126] border border-white/[0.08] focus:border-[#8265FF] rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none transition-all"
              >
                <option value="developer">Developer</option>
                <option value="admin">Administrator</option>
                <option value="analyst">Analyst</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-6 py-3 px-4 rounded-xl font-semibold text-sm bg-gradient-to-r from-[#8265FF] to-[#A855F7] hover:from-[#7152f8] hover:to-[#9333ea] text-white shadow-lg shadow-[#8265FF]/25 hover:shadow-[#8265FF]/40 disabled:opacity-50 transition-all duration-200 flex items-center justify-center gap-2"
          >
            {loading ? (
              <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <>
                <span>{isRegister ? "Create Account" : "Sign In"}</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        <div className="mt-6 pt-6 border-t border-white/[0.06] text-center">
          <button
            type="button"
            onClick={() => {
              setIsRegister(!isRegister);
              setError(null);
              setSuccess(null);
            }}
            className="text-xs text-[#8265FF] hover:text-[#A855F7] font-medium transition-colors"
          >
            {isRegister
              ? "Already have an account? Sign In"
              : "Don't have an account yet? Register here"}
          </button>
        </div>
      </div>
    </div>
  );
}
