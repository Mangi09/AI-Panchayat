"""
AI Panchayat – FastAPI Application  (BUGFIX v2.1)
──────────────────────────────────────────────────
Bugs fixed:
  Bug 3 – Fragile CSV parsing → _parse_csv() with chardet + sep=None
  Bug 4 – Hardcoded column defaults → target_col/sensitive_col are now
           required Query params on upload endpoints
New:
  POST /api/columns – returns columns + sample for the frontend column-picker
"""

import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal

try:
    import chardet
    _CHARDET = True
except ImportError:
    _CHARDET = False

from app.core.ml_engine import run_audit, run_mitigated_audit
from app.core.gemini_engine import generate_simulation_report
from app.core.data_generator import get_test_dataset, AVAILABLE_DATASETS

app = FastAPI(title="AI Panchayat – Bias Auditing API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MitigationMethod = Literal["reweighing", "exponentiated_gradient", "threshold_optimizer"]


# ── Bug 3 fix: robust CSV parser ──────────────────────────────────────────────

def _parse_csv(contents: bytes) -> pd.DataFrame:
    """
    Parse raw CSV bytes handling:
      • Unknown/mixed encodings — chardet detection + cascade fallback
      • BOM markers             — utf-8-sig strips BOM automatically
      • Non-comma delimiters    — sep=None + python engine auto-detects
                                  semicolons, tabs, pipes, etc.
      • Whitespace in column names — stripped after read
    Raises HTTPException(400) with a descriptive message on failure.
    """
    encoding = "utf-8"
    if _CHARDET:
        detected = chardet.detect(contents)
        if detected and detected.get("confidence", 0) >= 0.7:
            encoding = detected.get("encoding") or "utf-8"

    # Try encodings in cascade; latin-1 is a byte-safe last resort
    candidates = list(dict.fromkeys([encoding, "utf-8-sig", "utf-8", "latin-1", "cp1252"]))
    last_exc = None
    for enc in candidates:
        try:
            text = contents.decode(enc)
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            df.columns = df.columns.str.strip()
            if df.empty or df.shape[1] < 2:
                raise ValueError("CSV must contain at least two columns.")
            return df
        except Exception as exc:
            last_exc = exc
    raise HTTPException(
        status_code=400,
        detail=(
            f"Could not parse the uploaded file. "
            f"Ensure it is a valid CSV (comma or semicolon separated). "
            f"Last error: {last_exc}"
        ),
    )


# ── Bug 4 fix: descriptive column validation ──────────────────────────────────

_LABEL_HINTS = {
    "hired", "approved", "outcome", "label", "decision", "result",
    "target", "y", "class", "admitted", "priority_care_given", "loan_approved",
}
_SENS_HINTS = {
    "gender", "sex", "race", "ethnicity", "age", "age_group",
    "nationality", "religion", "disability", "zip_code_zone", "sensitive",
}


def _assert_columns(df: pd.DataFrame, target_col: str, sensitive_col: str):
    cols = list(df.columns)
    if target_col not in cols:
        suggestions = [c for c in cols if c.lower() in _LABEL_HINTS]
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found.{hint} Available columns: {cols}",
        )
    if sensitive_col not in cols:
        suggestions = [c for c in cols if c.lower() in _SENS_HINTS]
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise HTTPException(
            status_code=400,
            detail=f"Sensitive column '{sensitive_col}' not found.{hint} Available columns: {cols}",
        )


def _assert_binary_target(df: pd.DataFrame, target_col: str):
    n_unique = df[target_col].nunique()
    if n_unique != 2:
        vals = df[target_col].value_counts().head(5).index.tolist()
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_col}' must be binary (exactly 2 unique values). "
                f"Found {n_unique} unique values: {vals}. "
                f"Please binarise this column before uploading."
            ),
        )


def _load_test_dataset(name: str) -> dict:
    try:
        return get_test_dataset(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "AI Panchayat",
        "status": "online",
        "available_test_datasets": AVAILABLE_DATASETS,
        "mitigation_methods": ["reweighing", "exponentiated_gradient", "threshold_optimizer"],
    }


@app.post("/api/columns")
async def get_columns(file: UploadFile = File(...)):
    """
    Return column names + a 3-row preview from an uploaded CSV.
    The frontend uses this to render the column-picker step BEFORE
    calling /api/audit, so the user can specify which columns are
    the target and the sensitive attribute.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    df = _parse_csv(contents)
    sample = df.head(3).fillna("").astype(str).to_dict(orient="records")
    cols = list(df.columns)

    # Pre-select sensible UI defaults so common CSVs auto-fill correctly
    suggested_target    = next((c for c in cols if c.lower() in _LABEL_HINTS), None)
    suggested_sensitive = next((c for c in cols if c.lower() in _SENS_HINTS), None)

    return {
        "columns": cols,
        "sample_rows": sample,
        "total_rows": len(df),
        "suggested_target_col": suggested_target,
        "suggested_sensitive_col": suggested_sensitive,
    }


@app.post("/api/audit")
async def audit_csv(
    file: UploadFile = File(...),
    # BUG 4 FIX: required Query params — no more silent "target"/"sensitive" defaults
    target_col:    str = Query(..., description="Name of the binary target column"),
    sensitive_col: str = Query(..., description="Name of the protected-attribute column"),
):
    """Upload a CSV, run the ML bias audit, and get a Gemini debate report."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    df = _parse_csv(contents)          # BUG 3 FIX: robust parsing

    _assert_columns(df, target_col, sensitive_col)
    _assert_binary_target(df, target_col)

    metrics    = run_audit(df, target_col, sensitive_col)
    simulation = generate_simulation_report(metrics)
    return {"metrics": metrics, "simulation": simulation}


@app.get("/api/test_audit/{dataset_name}")
async def test_audit(dataset_name: str):
    """Run a bias audit on one of the pre-built test datasets."""
    info = _load_test_dataset(dataset_name)
    metrics    = run_audit(info["dataframe"], info["target_column"], info["sensitive_column"])
    simulation = generate_simulation_report(metrics)
    return {"metrics": metrics, "simulation": simulation}


@app.get("/api/datasets")
async def list_datasets():
    return {"datasets": AVAILABLE_DATASETS}


@app.get("/api/mitigate/{dataset_name}")
async def mitigate_test_dataset(
    dataset_name: str,
    method: MitigationMethod = Query(default="exponentiated_gradient"),
):
    info = _load_test_dataset(dataset_name)
    return run_mitigated_audit(
        info["dataframe"], info["target_column"], info["sensitive_column"], method=method
    )


@app.post("/api/mitigate")
async def mitigate_csv(
    file: UploadFile = File(...),
    target_col:    str = Query(..., description="Name of the binary target column"),
    sensitive_col: str = Query(..., description="Name of the protected-attribute column"),
    method: MitigationMethod = Query(default="exponentiated_gradient"),
):
    """Upload a CSV and apply bias mitigation."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    df = _parse_csv(contents)

    _assert_columns(df, target_col, sensitive_col)
    _assert_binary_target(df, target_col)

    return run_mitigated_audit(df, target_col, sensitive_col, method=method)
