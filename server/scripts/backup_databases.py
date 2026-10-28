#!/usr/bin/env python3
# server/scripts/backup_databases.py
"""
Online database backup and restoration utility for AutoVeyra platform.
Uses SQLite online backup API to take consistent zero-downtime snapshots of all databases.
"""

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVER_DIR / "data"
DEFAULT_BACKUP_ROOT = SERVER_DIR / "backups"

DB_NAMES = [
    "model_registry.db",
    "reports.db",
    "experiments.db",
    "tuning_history.db",
    "feature_history.db",
    "auth.db",
    "monitoring.db",
]


def backup_databases(target_dir: Path) -> None:
    """Performs consistent online snapshot of all active SQLite databases and model artifacts."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = target_dir / f"autoveyra_backup_{timestamp}"
    backup_path.mkdir(parents=True, exist_ok=True)
    db_backup_dir = backup_path / "databases"
    db_backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting AutoVeyra backup -> {backup_path.resolve()}")

    # 1. Online SQLite Backups
    backed_up_count = 0
    for db_name in DB_NAMES:
        source_db = DATA_DIR / db_name
        if source_db.exists():
            dest_db = db_backup_dir / db_name
            src_conn = sqlite3.connect(str(source_db))
            dst_conn = sqlite3.connect(str(dest_db))
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
                src_conn.close()
            print(f"  [+] Backed up SQLite database: {db_name}")
            backed_up_count += 1

    # 2. Model Artifacts Backup
    models_dir = DATA_DIR / "models"
    if models_dir.exists() and any(models_dir.iterdir()):
        dest_models_dir = backup_path / "models"
        shutil.copytree(str(models_dir), str(dest_models_dir), dirs_exist_ok=True)
        artifact_count = len(list(dest_models_dir.glob("*.joblib")))
        print(f"  [+] Backed up {artifact_count} model joblib artifacts.")

    print(f"[v] Backup successfully completed: {backed_up_count} databases stored in {backup_path.resolve()}\n")


def restore_backup(backup_path: Path) -> None:
    """Restores SQLite databases and model artifacts from a previously taken backup snapshot."""
    if not backup_path.exists() or not backup_path.is_dir():
        print(f"[!] Error: Backup directory {backup_path} does not exist.")
        sys.exit(1)

    print(f"[*] Restoring AutoVeyra state from -> {backup_path.resolve()}")
    db_backup_dir = backup_path / "databases"
    if db_backup_dir.exists():
        for db_name in DB_NAMES:
            src = db_backup_dir / db_name
            if src.exists():
                dst = DATA_DIR / db_name
                shutil.copy2(str(src), str(dst))
                print(f"  [+] Restored database: {db_name}")

    models_backup_dir = backup_path / "models"
    if models_backup_dir.exists():
        dst_models = DATA_DIR / "models"
        dst_models.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(models_backup_dir), str(dst_models), dirs_exist_ok=True)
        print(f"  [+] Restored model artifacts into {dst_models.resolve()}")

    print("[v] Restoration completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="AutoVeyra Database & Artifact Backup Utility")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_ROOT, help="Directory to store backup snapshots")
    parser.add_argument("--restore", type=Path, default=None, help="Path to backup snapshot directory to restore")
    args = parser.parse_args()

    if args.restore:
        restore_backup(args.restore)
    else:
        backup_databases(args.backup_dir)


if __name__ == "__main__":
    main()
