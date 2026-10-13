# SupplyGuard: Warranty Parts Forecasting & Inventory Optimization

An end-to-end, reproducible decision-support project using **synthetic** warranty-parts demand and supplier records. Predict future parts demand, classify supplier delays, and recommend purchase quantities under budget, storage and service-level constraints.

## Architecture
Synthetic generation -> chronological feature engineering -> time-based holdout -> forecasting + delay classification -> optimization -> FastAPI -> MLflow.

**No real warranty/customer data is included.** Outputs are planning recommendations, not automatic procurement orders.

## Quick start
Python 3.10+ recommended.
```bash
pip install -r requirements.txt
python -m supplyguard.data
python -m supplyguard.train
python -m supplyguard.optimize
uvicorn supplyguard.api:app --reload
pytest -q
```
Open http://127.0.0.1:8000/docs. The training command writes artifacts/metrics.json, artifacts/forecasts.csv and joblib models. Optimization writes artifacts/recommendations.csv. Synthetic data generation is deterministic. MLflow records metrics and parameters locally in mlruns/.

## Methods
- Demand: lag 1/2/3, rolling historical 3-month mean, seasonality, part and region; train/validation/test split by month (no random row split). Seasonal-naive baseline compared with XGBoost; WAPE on held-out demand.
- Supplier delay: historical lead-time and order features; XGBoost classifier, held-out ROC-AUC when both classes exist.
- OR-Tools CBC: minimize purchase + shortage penalty + holding cost, subject to budget, storage capacity, order upper bounds, and minimum stock targets. Reports infeasible constraints rather than inventing a solution.
- API: health, forecast results and optimization results.
- Synthetic demonstration only; cost and service assumptions are editable in supplyguard/optimize.py.

## Responsible use
Do not use generated numbers as empirical results. Inspect metrics.json after running and report actual test performance. Forecasting and supplier-delay predictions are experimental, and the optimizer uses simplified deterministic demand and cost assumptions. No production deployment or real-data validation is claimed.
