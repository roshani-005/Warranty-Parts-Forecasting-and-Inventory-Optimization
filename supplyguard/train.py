"""Train/evaluate on time-separated synthetic observations."""
import json
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
from .data import save
from .features import FEATURES, make_features, chronological_split

def wape(y, pred):
    denom = np.abs(np.asarray(y)).sum()
    return float(np.abs(np.asarray(y)-np.asarray(pred)).sum()/denom) if denom else 0.0

def run():
    raw = save()
    frame = make_features(raw)
    train, val, test = chronological_split(frame)
    Path("artifacts").mkdir(exist_ok=True)
    params = dict(n_estimators=140, max_depth=3, learning_rate=.05, random_state=42, n_jobs=1)
    baseline_val = wape(val.demand, val.lag_1)
    model = XGBRegressor(**params)
    model.fit(train[FEATURES], train.demand)
    val_wape = wape(val.demand, np.maximum(0, model.predict(val[FEATURES])))
    # Choose model on validation only; refit on train+validation for final test.
    use_model = val_wape < baseline_val
    combined = pd.concat([train,val])
    model.fit(combined[FEATURES], combined.demand)
    pred = np.maximum(0, model.predict(test[FEATURES])) if use_model else test.lag_1.to_numpy()
    forecast = test[["month","part","region","demand","unit_cost","unit_volume"]].copy()
    forecast["prediction"] = pred
    forecast.to_csv("artifacts/forecasts.csv", index=False)
    metrics = {"validation_baseline_wape":baseline_val,"validation_xgboost_wape":val_wape,"selected_model":"xgboost" if use_model else "lag_1_baseline","test_wape":wape(test.demand,pred),"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test)}
    # Supplier delay is a separate task; features available at order placement only.
    delay_features = ["promised_days","unit_cost","unit_volume","part_code","region_code","month_num"]
    delay_model = XGBClassifier(**params, eval_metric="logloss")
    if combined.delayed.nunique() == 2:
        delay_model.fit(combined[delay_features], combined.delayed)
        metrics["delay_test_roc_auc"] = float(roc_auc_score(test.delayed, delay_model.predict_proba(test[delay_features])[:,1])) if test.delayed.nunique()==2 else None
        joblib.dump(delay_model, "artifacts/delay_model.joblib")
    else:
        metrics["delay_test_roc_auc"] = None
    joblib.dump(model, "artifacts/demand_model.joblib")
    with mlflow.start_run(run_name="supplyguard-synthetic"):
        mlflow.log_params(params)
        mlflow.log_param("selected_model", metrics["selected_model"])
        mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float)) and v is not None})
    Path("artifacts/metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

if __name__ == "__main__":
    print(json.dumps(run(),indent=2))
