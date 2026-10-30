# AutoVeyra 2.0 — Free Public Deployment Guide (Render)

This guide walks you through deploying **AutoVeyra 2.0** on **Render's 100% Free Tier** using Render Blueprints (`render.yaml`).

With this setup:
* **Frontend**: Hosted on Render Free Static Site (`autoveyra-web`) with automatic HTTPS and SPA routing fallback.
* **Backend**: Hosted on Render Free Python Web Service (`autoveyra-api`) with automatic HTTPS and `/health` monitoring.
* **Automatic Deployments**: Every push to your GitHub `main` branch automatically triggers zero-downtime redeployments.
* **Cost**: **$0.00 / month (Absolutely Free)**.

---

## Architecture Overview

```
                      +------------------------------------------+
                      |               Web Browser                |
                      +--------------------+---------------------+
                                           |
               +---------------------------+---------------------------+
               | (HTTPS static assets)                                 | (HTTPS REST API)
               v                                                       v
  +--------------------------+                            +--------------------------+
  | Render Static Site       |                            | Render Web Service       |
  | `autoveyra-web`          |                            | `autoveyra-api`          |
  | React 19 + Vite 6        |                            | FastAPI + Uvicorn        |
  | SPA Rewrite -> index.html|                            | Python 3.11 Free Tier    |
  | VITE_API_BASE_URL        |                            | Port: $PORT (0.0.0.0)    |
  +--------------------------+                            +-------------+------------+
                                                                        |
                                                          +-------------+------------+
                                                          | Ephemeral Storage &      |
                                                          | In-Memory ML Pipelines   |
                                                          | SQLite WAL Databases     |
                                                          +--------------------------+
```

---

## Step-by-Step Deployment (Under 10 Simple Steps)

### Step 1: Sign Up for Render (Free)
1. Go to [https://render.com](https://render.com).
2. Click **Sign Up** and choose **Continue with GitHub**.
3. Authorize Render to access your GitHub repositories (or select only the `auto-ml` repository).

### Step 2: Open Render Blueprints
1. In the Render Dashboard ([https://dashboard.render.com](https://dashboard.render.com)), navigate to the top-left menu or click the **New +** button in the top navigation bar.
2. Select **Blueprint** from the dropdown menu.

### Step 3: Connect your Repository
1. In the **Connect a repository** list, locate your repository:
   `paliwalpriyanjal-hash/auto-ml` (or search for `auto-ml`).
2. Click **Connect**.

### Step 4: Configure Blueprint Instance
1. Render will automatically detect `render.yaml` in the root of your repository.
2. **Blueprint Name**: Enter `autoveyra` (or leave default).
3. **Branch**: Ensure `main` is selected.
4. Render displays the two planned services:
   * `autoveyra-web` (Static Site, Free)
   * `autoveyra-api` (Web Service, Python, Free)
5. Click **Apply**.

### Step 5: Service Provisioning & Initial Build
1. Render will begin deploying both services in parallel:
   * **Backend (`autoveyra-api`)**:
     * Installs dependencies via `pip install --upgrade pip && pip install -r requirements.txt`.
     * Runs startup hook initializing all runtime directories and SQLite schemas.
     * Binds to `0.0.0.0:$PORT`.
     * Health check passes at `/health`.
   * **Frontend (`autoveyra-web`)**:
     * Installs Node packages via `npm ci`.
     * Injects `VITE_API_BASE_URL=https://autoveyra-api.onrender.com` during the Vite build.
     * Publishes `dist/` directory.
     * Rewrites `/*` to `/index.html` for React Router SPA navigation.
2. Wait 3–5 minutes for the initial builds to complete.

---

## Verifying Public Endpoints

Once deployment finishes, Render provides default `.onrender.com` subdomains:

### 1. Backend Verification
* **Base URL**: `https://autoveyra-api.onrender.com`
* **Health Check**: Open `https://autoveyra-api.onrender.com/health` in your browser.
  Expected JSON response:
  ```json
  {
    "status": "healthy",
    "app": "AutoVeyra Tabular AutoML Platform",
    "version": "2.0-PRO",
    "environment": "production"
  }
  ```
* **Interactive OpenAPI Docs**: Open `https://autoveyra-api.onrender.com/docs` to verify all interactive endpoints.
* **Deep Readiness Probe**: Open `https://autoveyra-api.onrender.com/ready` to verify SQLite read/write and storage access.

### 2. Frontend Verification
* **Public Web URL**: `https://autoveyra-web.onrender.com`
* Open the URL in your browser.
* Look at the top navigation bar:
  * Green **"Backend Online"** indicator should show that the frontend successfully communicated with the deployed backend.
  * The dismissable **Demo Environment** banner will inform visitors of free-tier ephemeral storage.

---

## Custom Service Names or Subdomains

If the default service name `autoveyra-api` is already claimed on Render and Render assigns a unique suffix (e.g. `autoveyra-api-xyz`), synchronize the two environment variables:

1. **In `autoveyra-web` Environment Variables**:
   * Navigate to `autoveyra-web` -> **Environment**.
   * Update `VITE_API_BASE_URL` to `https://<your-actual-backend-name>.onrender.com`.
   * Update `VITE_API_URL` to `https://<your-actual-backend-name>.onrender.com`.
   * Click **Save Changes** and trigger a **Manual Deploy -> Clear build cache & deploy** (Vite embeds environment variables at build time).

2. **In `autoveyra-api` Environment Variables**:
   * Navigate to `autoveyra-api` -> **Environment**.
   * Update `AUTOVEYRA_ALLOWED_ORIGINS` to `https://<your-actual-frontend-name>.onrender.com,http://localhost:5173`.
   * Click **Save Changes**. The backend restarts immediately without needing a rebuild.

---

## Environment Variables Reference

### Backend (`autoveyra-api`)

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `AUTOVEYRA_ENV` | `production` | Deployment environment identifier |
| `AUTOVEYRA_AUTH_ENABLED` | `false` | `false` allows instant guest access for demonstrations; `true` enables strict JWT authentication |
| `AUTOVEYRA_JWT_SECRET` | Auto-generated | Cryptographically secure random secret generated by Render |
| `AUTOVEYRA_ALLOWED_ORIGINS` | Comma-separated | Allowed CORS origins (e.g. `https://autoveyra-web.onrender.com,http://localhost:5173`) |
| `AUTOVEYRA_MAX_UPLOAD_BYTES`| `26214400` (25 MB) | Maximum dataset upload size limit |
| `AUTOVEYRA_MAX_DATASET_ROWS` | `50000` | Free-tier RAM safety ceiling for dataset row processing |
| `AUTOVEYRA_MAX_DATASET_COLS` | `200` | Free-tier column width ceiling |
| `AUTOVEYRA_MAX_TUNING_TRIALS`| `25` | Maximum Optuna Bayesian trials on free compute |
| `AUTOVEYRA_MAX_EXPLAIN_SAMPLES`| `200` | Maximum permutation / SHAP evaluation sample size |
| `PYTHON_VERSION` | `3.11.9` | Optimized Python runtime on Linux |

### Optional Admin Credentials (When `AUTOVEYRA_AUTH_ENABLED=true`)

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `AUTOVEYRA_ADMIN_USERNAME` | `admin` | Bootstrap administrator username |
| `AUTOVEYRA_ADMIN_PASSWORD` | `AdminPassword123!` | Bootstrap administrator password |

### Frontend (`autoveyra-web`)

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `https://autoveyra-api.onrender.com` | Base URL of deployed backend (evaluated during Vite build) |
| `VITE_API_URL` | `https://autoveyra-api.onrender.com` | Fallback backend URL for legacy API helpers |

---

## Free-Tier Operational Characteristics & Limitations

> [!NOTE]
> Render's free tier provides high reliability and free SSL certificates, but has specific cloud constraints designed for non-production showcase environments:

1. **Instance Hibernation (Cold Starts)**:
   * On Render's Free tier, the web service automatically spins down to 0 instances after **15 minutes of inactivity**.
   * The first incoming HTTP request wakes up the container, taking **45–60 seconds** (cold start). Subsequent requests respond in sub-second time.
   * *Tip*: In college demos or portfolio presentations, open the website 1 minute before your demonstration to ensure the backend is warm.

2. **Ephemeral Filesystem (Reset on Redeploy)**:
   * Free-tier container storage is non-persistent.
   * Uploaded test CSVs, trained model `.joblib` files, generated PDF/ZIP reports, and SQLite tables reset when the service is redeployed or spun down.
   * AutoVeyra 2.0 includes automated startup hooks (`init_application_state`) to regenerate all tables and directories automatically whenever a container boots up.

3. **RAM Ceiling (512 MB Memory Limit)**:
   * Free instances have 512 MB of RAM.
   * AutoVeyra 2.0 has been hardened to prevent OOM errors by:
     * Lazy-loading heavy packages (e.g. SHAP, numba) so initial boot uses ~80 MB RAM.
     * Capping default batch prediction rows to 2,500 and tuning trials to 25.
     * Setting maximum demo dataset rows to 50,000.
   * *Do not attempt to train massive deep-learning models or multi-gigabyte datasets on the free plan.*

---

## Continuous Deployment (Automatic Updates)

Because your Blueprint is linked to branch `main`:
1. Whenever you run:
   ```bash
   git push origin main
   ```
2. Render detects the commit and triggers an automatic deployment.
3. If only frontend files changed, Render redeploys `autoveyra-web`. If backend files changed, Render redeploys `autoveyra-api`.
