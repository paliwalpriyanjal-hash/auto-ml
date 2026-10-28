# server/app/utils/security.py
"""
Input validation, upload security, and sanitization utilities for AutoVeyra.
Guarantees bounded resource consumption, path-traversal isolation, and encoding resilience.
"""

import io
import re
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import HTTPException, UploadFile

from app.config import settings


async def safe_read_csv(
    file: UploadFile,
    max_bytes: Optional[int] = None,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely reads an uploaded tabular CSV file with bounded memory, size limits,
    encoding fallbacks, and structure validation.
    """
    limit_bytes = max_bytes or settings.MAX_UPLOAD_SIZE_BYTES
    limit_rows = max_rows or settings.MAX_DATASET_ROWS
    limit_cols = max_cols or settings.MAX_DATASET_COLS

    # 1. Read bytes in chunks to enforce size limit without buffering huge payloads in memory
    chunk_size = 64 * 1024  # 64 KB chunks
    total_bytes = 0
    buffer = io.BytesIO()

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > limit_bytes:
            mb = limit_bytes // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed limit of {mb}MB.",
            )
        buffer.write(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty.",
        )

    # 2. Resilient Decoding (UTF-8 primary, Latin-1 fallback)
    raw_bytes = buffer.getvalue()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("latin-1")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode CSV text. File must be UTF-8 or Latin-1 encoded: {exc}",
            )

    # 3. Parse CSV safely
    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file contains no readable data.")
    except pd.errors.ParserError as parse_err:
        raise HTTPException(
            status_code=400,
            detail=f"Malformed CSV formatting: {parse_err}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV dataset: {exc}",
        )

    if df.empty or len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV file contains zero data rows.",
        )

    # 4. Dimension & Bounds Checks
    if len(df) > limit_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset exceeds maximum allowed row limit of {limit_rows:,} rows. Found {len(df):,} rows.",
        )

    if len(df.columns) > limit_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset exceeds maximum allowed column limit of {limit_cols:,} columns. Found {len(df.columns):,} columns.",
        )

    # 5. Clean & Validate Headers
    cleaned_headers = [str(col).strip() for col in df.columns]
    if any(not name for name in cleaned_headers):
        raise HTTPException(
            status_code=400,
            detail="CSV contains an empty column header. All columns must have valid names.",
        )

    if len(set(cleaned_headers)) != len(cleaned_headers):
        raise HTTPException(
            status_code=400,
            detail="CSV contains duplicate column headers. Each column name must be unique.",
        )

    df.columns = cleaned_headers
    return df


def validate_prediction_payload(rows: Any, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Validates batch prediction payload type, structure, and bounded row count.
    """
    limit_rows = max_rows or settings.MAX_PREDICTION_ROWS

    if not isinstance(rows, list):
        raise HTTPException(
            status_code=400,
            detail="Prediction input payload must be a JSON array of row objects.",
        )

    if len(rows) == 0:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one row for prediction.",
        )

    if len(rows) > limit_rows:
        raise HTTPException(
            status_code=413,
            detail=f"Batch prediction exceeds maximum allowed limit of {limit_rows:,} rows (received {len(rows):,} rows).",
        )

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Row item at index {idx} is invalid. Each row must be a JSON object of key-value feature pairs.",
            )

    return rows


def sanitize_filename(filename: Optional[str]) -> str:
    """
    Strips directory traversal sequences and unsafe characters from filenames.
    """
    if not filename:
        return "dataset.csv"
    base = filename.replace("\\", "/").split("/")[-1]
    safe = re.sub(r"[^\w\-.]", "_", base)
    return safe or "dataset.csv"
