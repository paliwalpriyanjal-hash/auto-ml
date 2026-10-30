// client/src/utils/workspaceContext.js
// Workspace state management and continuity helpers for AutoVeyra Phase 6.

const STORAGE_KEY = "autoveyra_workspace_context";
const MODE_KEY = "autoveyra_experience_mode";
const ONBOARDING_KEY = "autoveyra_onboarding_completed";

export const DEFAULT_WORKSPACE_STATE = {
  project_id: "",
  name: "My First Machine Learning Project",
  experience_mode: "guided", // 'guided' | 'auto' | 'advanced'
  current_step: 1,
  completed_steps: [],
  dataset_id: "",
  dataset_name: "",
  row_count: 0,
  column_count: 0,
  columns: [],
  target_column: "",
  task_type: "classification",
  active_runs: {},
  preview_data: [],
  intelligence: null,
  clean_summary: null,
  trained_models: [],
  champion_model: null,
  explainability: null,
  report: null,
};

export function getExperienceMode() {
  try {
    return localStorage.getItem(MODE_KEY) || "guided";
  } catch {
    return "guided";
  }
}

export function setExperienceMode(mode) {
  try {
    localStorage.setItem(MODE_KEY, mode);
  } catch {
    // ignore
  }
}

export function isOnboardingCompleted() {
  try {
    return localStorage.getItem(ONBOARDING_KEY) === "true";
  } catch {
    return false;
  }
}

export function setOnboardingCompleted(completed = true) {
  try {
    localStorage.setItem(ONBOARDING_KEY, completed ? "true" : "false");
  } catch {
    // ignore
  }
}

export function loadWorkspaceState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_WORKSPACE_STATE, experience_mode: getExperienceMode() };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_WORKSPACE_STATE, ...parsed, experience_mode: getExperienceMode() };
  } catch (err) {
    console.warn("Failed to load workspace state from localStorage:", err);
    return { ...DEFAULT_WORKSPACE_STATE, experience_mode: getExperienceMode() };
  }
}

export function saveWorkspaceState(state) {
  try {
    // Avoid saving massive raw arrays in localStorage
    const safeCopy = { ...state };
    if (safeCopy.preview_data && safeCopy.preview_data.length > 20) {
      safeCopy.preview_data = safeCopy.preview_data.slice(0, 20);
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(safeCopy));
  } catch (err) {
    console.warn("Failed to persist workspace state:", err);
  }
}

export function clearWorkspaceState() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
