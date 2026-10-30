// client/src/utils/api.js
// Centralized API configuration and production utility helpers

export const DEFAULT_API_URL = "http://127.0.0.1:8000";

export function getApiBase() {
  const configured = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL;
  return (configured || DEFAULT_API_URL).replace(/\/+$/, "");
}

export function formatMetric(val, digits = 4) {
  if (val === null || val === undefined || val === "") return "N/A";
  const num = Number(val);
  return Number.isFinite(num) ? num.toFixed(digits) : String(val);
}

export function formatPercent(val, digits = 1) {
  if (val === null || val === undefined || val === "") return "N/A";
  const num = Number(val);
  return Number.isFinite(num) ? `${(num * 100).toFixed(digits)}%` : "N/A";
}

export function getAuthToken() {
  try {
    return localStorage.getItem("autoveyra_token") || "";
  } catch {
    return "";
  }
}

export function setAuthToken(token) {
  try {
    if (token) {
      localStorage.setItem("autoveyra_token", token);
    } else {
      localStorage.removeItem("autoveyra_token");
    }
  } catch {
    // LocalStorage access might fail in restricted iframe environments
  }
}

/**
 * Fetch wrapper with configurable AbortController timeout, Bearer token injection, and standard header propagation.
 */
export async function fetchWithTimeout(url, options = {}, timeoutMs = 60000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  const token = getAuthToken();
  const headers = new Headers(options.headers || {});
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });
    return response;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`Request timed out after ${timeoutMs / 1000} seconds.`);
    }
    throw err;
  } finally {
    clearTimeout(id);
  }
}

/**
 * Robust error extractor parsing FastAPI structured error responses, JSON details, or fallback message.
 */
export async function extractErrorMessage(response, defaultMsg = "Request failed.") {
  if (!response) return defaultMsg;
  try {
    const data = await response.json();
    if (typeof data.detail === "string") {
      return data.detail;
    }
    if (Array.isArray(data.detail)) {
      return data.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
    }
    if (typeof data.message === "string") {
      return data.message;
    }
  } catch {
    // If response body is not JSON or already read
    if (response.statusText) {
      return `${defaultMsg} (${response.status}: ${response.statusText})`;
    }
  }
  return defaultMsg;
}
