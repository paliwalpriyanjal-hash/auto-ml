# AutoVeyra — Production Deployment & Operations Guide

## 1. Overview & Trust Architecture

AutoVeyra is an enterprise tabular AutoML platform designed for containerized single-tenant and private VPC deployments. This document provides step-by-step instructions for container deployment, reverse proxy TLS termination, persistent volume management, zero-downtime SQLite backups, and authentication enforcement.

---

## 2. Containerized Deployment Architecture

AutoVeyra provides a hardened, non-root Docker runtime with isolated data and artifact storage:

### Docker Architecture
- **Base Image**: `python:3.11-slim` (minimal attack surface, free of unnecessary compilers/tools).
- **Process Security**: Runs under unprivileged user `autoveyra` (`UID 10001:GID 10001`).
- **Storage Isolation**: Dedicated volume mount points for databases, serialized models, and generated reports.
- **Health Probing**: Integrated Docker `HEALTHCHECK` exercising `/health` with 30s intervals.

### Quick Start with Docker Compose

```bash
# 1. Clone repository and navigate to root
cd AutoML-Studio-main

# 2. Configure environment secrets
cp .env.example .env   # Or set environment variables

# 3. Build and launch services in detached mode
docker-compose up -d --build

# 4. Verify running containers and health checks
docker-compose ps
```

### Volume Persistence Layout

| Volume Name | Container Path | Purpose | Backup Strategy |
|:---|:---|:---|:---|
| `autoveyra_data` | `/app/data` | SQLite databases (`autoveyra.db`, `auth.db`, `monitoring.db`) | Hot online backup via `backup_databases.py` |
| `autoveyra_models` | `/app/models` | Serialized scikit-learn/joblib model artifacts | Tarball snapshot with checksum validation |
| `autoveyra_reports` | `/app/reports` | Exported PDF, HTML, and JSON report files | Incremental filesystem backup |

---

## 3. Environment Configuration & Security Hardening

Configure runtime behavior via environment variables in `.env` or container orchestration manifests (Kubernetes / Docker Compose):

```ini
# --- Environment & Security ---
AUTOVEYRA_ENV=production
DEBUG=false
AUTOVEYRA_AUTH_ENABLED=true
AUTOVEYRA_JWT_SECRET=CHANGE_THIS_TO_A_64_CHAR_CRYPTOGRAPHIC_SECRET
AUTOVEYRA_JWT_ALGORITHM=HS256
AUTOVEYRA_JWT_EXPIRATION_HOURS=24

# --- Bootstrapping Admin Account ---
AUTOVEYRA_ADMIN_USERNAME=enterprise_admin
AUTOVEYRA_ADMIN_PASSWORD=SuperSecureProductionPassword2026!

# --- Model Serving & Concurrency ---
AUTOVEYRA_MODEL_CACHE_SIZE=16
AUTOVEYRA_MAX_CONCURRENT_WORKLOADS=2
AUTOVEYRA_MAX_MONITORING_RECORDS=50000

# --- Resource & Input Guardrails ---
AUTOVEYRA_MAX_UPLOAD_BYTES=52428800
AUTOVEYRA_MAX_DATASET_ROWS=1000000
AUTOVEYRA_MAX_DATASET_COLS=500
AUTOVEYRA_MAX_PREDICTION_ROWS=50000

# --- CORS & Network ---
ALLOWED_ORIGINS=https://autoveyra.internal.company.com
```

> [!WARNING]
> In production environments, never leave `AUTOVEYRA_JWT_SECRET` at its default value. Always configure an external cryptographically random secret of at least 256 bits (`openssl rand -hex 32`).

---

## 4. Production Reverse Proxy Configuration (TLS & Reverse Proxy)

AutoVeyra backend services should sit behind an enterprise reverse proxy (such as Nginx, Caddy, or AWS ALB) terminating HTTPS, enforcing HTTP Strict Transport Security (HSTS), and restricting rate limits.

### Sample Nginx Configuration (`/etc/nginx/conf.d/autoveyra.conf`)

```nginx
upstream autoveyra_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name autoveyra.internal.company.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name autoveyra.internal.company.com;

    ssl_certificate /etc/ssl/certs/autoveyra.crt;
    ssl_certificate_key /etc/ssl/private/autoveyra.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Static Frontend SPA
    root /var/www/autoveyra/client/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # API Gateway Reverse Proxy
    location ~ ^/(auth|models|monitoring|reports|train|compare|clean|experiments|features|tune|health|readiness) {
        proxy_pass http://autoveyra_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;

        # Upload limits
        client_max_body_size 50M;
        proxy_read_timeout 300s;
    }
}
```

---

## 5. Zero-Downtime Database Backup & Restoration

AutoVeyra uses an automated SQLite online hot backup mechanism leveraging Python's `sqlite3.Connection.backup()`. This copies pages directly while queries continue running, preventing SQLite `database is locked` errors.

### Performing a Hot Backup

Run the automated backup script manually or via cron / systemd timer:

```bash
python server/scripts/backup_databases.py --backup-dir /mnt/backups/autoveyra
```

The script outputs a timestamped snapshot directory:
- `autoveyra_<timestamp>.db` (models, experiments, tuning records)
- `auth_<timestamp>.db` (users, credentials, revoked token blocklist)
- `monitoring_<timestamp>.db` (operational inference logs, latency metrics)
- `models_<timestamp>.tar.gz` (all serialized model artifacts)
- `manifest_<timestamp>.json` (SHA-256 integrity checksums for all files)

### Automated Cron Job Example

```cron
# Daily backup at 02:00 AM UTC
0 2 * * * cd /opt/autoveyra && /opt/autoveyra/venv/bin/python server/scripts/backup_databases.py >> /var/log/autoveyra_backup.log 2>&1
```

### Performing Disaster Recovery Restoration

```bash
# To safely restore from a verified snapshot:
python server/scripts/backup_databases.py --restore /mnt/backups/autoveyra/snapshot_20261029_020000
```

---

## 6. Monitoring & Health Probing

AutoVeyra exposes three operational health and readiness probe endpoints:

| Endpoint | HTTP Status | Description |
|:---|:---|:---|
| `GET /health` | 200 OK | Liveness probe: verifies process is alive and responsive. |
| `GET /readiness` | 200 OK / 503 | Readiness probe: executes active `SELECT 1` queries against `autoveyra.db`, `auth.db`, and `monitoring.db`. |
| `GET /monitoring/stats` | 200 OK | Operational metrics: total volume, error rate, p50, p90, p99 latency. |
| `GET /monitoring/models` | 200 OK | Actively loaded models in in-memory LRU cache and capacity status. |
| `GET /monitoring/drift/{id}` | 200 OK | Feature-level summary comparison tracking covariate input shifts. |

---

## 7. Single-Tenant Trust Boundary Disclosure

> [!IMPORTANT]
> AutoVeyra is engineered as a secure **single-tenant or private VPC enterprise platform**. While it implements strict authentication, cryptographic token revocation, joblib path sandboxing, CSV upload limits, and bounded telemetry, it does not sandbox third-party Python code execution in separate multi-tenant hypervisors. All organizations operating AutoVeyra must deploy it within an authenticated private intranet, single-tenant VPC, or containerized perimeter.
