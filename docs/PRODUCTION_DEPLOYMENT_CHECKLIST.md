# AutoVeyra 2.0-PRO — Production Deployment & Operator Runbook

This guide details the pre-flight requirements, configuration steps, and operational procedures for deploying AutoVeyra 2.0-PRO in production.

---

## 1. System Requirements

### Hardware Sizing
- **CPU**: Minimum 2 cores (4+ cores recommended for Optuna tuning).
- **RAM**: Minimum 4 GB (8 GB–16 GB recommended for 50k+ row datasets).
- **Storage**: Minimum 10 GB free persistent disk space (SSD recommended for model joblib and SQLite storage).

### Software Prerequisites
- **Python**: 3.12 or 3.13 (with `pip`, `venv`).
- **Node.js**: 20.x or 22.x (with `npm`).
- **Optional**: Docker 24+ and Docker Compose v2 (if running in container mode).

---

## 2. Configuration & Environment Variables

Copy `.env.example` to `.env` in the server root. Configure the following variables:

```bash
# General Settings
AUTOVEYRA_ENV=production
AUTOVEYRA_APP_NAME="AutoVeyra Tabular AutoML Platform"

# Security & Authentication (CRITICAL)
AUTOVEYRA_AUTH_ENABLED=true
AUTOVEYRA_JWT_SECRET="CHANGE_ME_TO_A_CRYPTOGRAPHICALLY_SECURE_32_CHAR_SECRET_KEY!!"
AUTOVEYRA_JWT_EXPIRATION_HOURS=24

# Ingestion Limits
AUTOVEYRA_MAX_UPLOAD_BYTES=26214400    # 25 MB
AUTOVEYRA_MAX_DATASET_ROWS=100000       # 100,000 rows
AUTOVEYRA_MAX_DATASET_COLS=500          # 500 features

# Serving & Tuning
AUTOVEYRA_MODEL_CACHE_SIZE=8
AUTOVEYRA_MAX_PREDICTION_ROWS=5000
AUTOVEYRA_MAX_TUNING_TRIALS=50

# CORS Allowed Origins (Comma-separated list)
AUTOVEYRA_ALLOWED_ORIGINS="https://autoveyra.yourdomain.com,http://localhost:5173"
```

---

## 3. Deployment Methods

### Method A: Docker Compose (Recommended for Containerized Hosts)

1. **Build and start containers**:
   ```bash
   docker compose up -d --build
   ```
2. **Verify container health**:
   ```bash
   docker compose ps
   curl -f http://localhost:8000/health
   curl -f http://localhost:8000/ready
   ```
3. **Inspect logs**:
   ```bash
   docker compose logs -f api
   ```

### Method B: Native Process Deployment (Non-Docker Hosts)

1. **Backend Service**:
   ```bash
   cd server
   python -m venv .venv
   # Windows: .venv\Scripts\activate | Linux: source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
   ```
2. **Frontend Service**:
   ```bash
   cd client
   npm install
   npm run build
   # Serve dist/ via Nginx, Caddy, or Node preview:
   npm run preview -- --host 0.0.0.0 --port 5173
   ```

---

## 4. Reverse Proxy & TLS Configuration (Nginx Example)

AutoVeyra should be placed behind a reverse proxy handling TLS termination:

```nginx
server {
    listen 443 ssl http2;
    server_name autoveyra.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/autoveyra.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/autoveyra.yourdomain.com/privkey.pem;

    client_max_body_size 30M;

    # Frontend Assets
    location / {
        root /var/www/autoveyra/client/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API Routing
    location ~ ^/(health|ready|upload|clean|preview|train|compare|export|experiments|models|tuning|features|reports|auth|monitoring|eda|assistant) {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

---

## 5. Routine Backup & Disaster Recovery Schedule

Configure a daily cron job or scheduled task to take an online hot snapshot using the built-in backup script:

```bash
# Daily backup snapshot at 02:00 UTC
0 2 * * * cd /app/server && python scripts/backup_databases.py --backup-dir /var/backups/autoveyra
```

To restore from a backup snapshot:
```bash
python scripts/backup_databases.py --restore /var/backups/autoveyra/autoveyra_backup_YYYYMMDD_HHMMSS
```
