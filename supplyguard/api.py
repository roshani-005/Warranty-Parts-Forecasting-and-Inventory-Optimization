"""Read-only demonstration API; generate artifacts with CLI first."""
from pathlib import Path
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
app=FastAPI(title="SupplyGuard",version="0.1.0")

@app.get("/health")
def health(): return {"status":"ok","data":"synthetic"}

@app.get("/metrics")
def metrics():
    p=Path("artifacts/metrics.json")
    if not p.exists(): raise HTTPException(404,"Run training first")
    return json.loads(p.read_text())

@app.get("/forecasts")
def forecasts():
    p=Path("artifacts/forecasts.csv")
    if not p.exists(): raise HTTPException(404,"Run training first")
    return pd.read_csv(p).to_dict(orient="records")

@app.get("/recommendations")
def recommendations():
    p=Path("artifacts/recommendations.csv")
    if not p.exists(): raise HTTPException(404,"Run optimization first")
    return pd.read_csv(p).to_dict(orient="records")
