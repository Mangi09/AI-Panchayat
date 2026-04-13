"""
AI Panchayat – FastAPI Application
Two endpoints: CSV upload audit and pre-built test dataset audit.
"""

import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.ml_engine import run_audit
from app.core.gemini_engine import generate_simulation_report
from app.core.data_generator import get_test_dataset, AVAILABLE_DATASETS

app = FastAPI(
    title="AI Panchayat – Bias Auditing API",
    version="1.0.0",
    description="Upload a CSV or pick a test dataset to run a Fairlearn bias audit "
                "and receive a multi-agent Gemini debate on the findings.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "AI Panchayat",
        "status": "online",
        "available_test_datasets": AVAILABLE_DATASETS,
    }


@app.post("/api/audit")
async def audit_csv(
    file: UploadFile = File(...),
    target_col: str = "target",
    sensitive_col: str = "sensitive",
):
    """Upload a CSV, run the ML bias audit, and get a Gemini debate report."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")

    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found. Columns: {list(df.columns)}")
    if sensitive_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Sensitive column '{sensitive_col}' not found. Columns: {list(df.columns)}")

    metrics = run_audit(df, target_col, sensitive_col)
    simulation = generate_simulation_report(metrics)

    return {"metrics": metrics, "simulation": simulation}


@app.get("/api/test_audit/{dataset_name}")
async def test_audit(dataset_name: str):
    """Run a bias audit on one of the pre-built test datasets."""
    try:
        df, target_col, sensitive_col = get_test_dataset(dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    metrics = run_audit(df, target_col, sensitive_col)
    simulation = generate_simulation_report(metrics)

    return {"metrics": metrics, "simulation": simulation}


@app.get("/api/datasets")
async def list_datasets():
    """List available test datasets."""
    return {"datasets": AVAILABLE_DATASETS}
