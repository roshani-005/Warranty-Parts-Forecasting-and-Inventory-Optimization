"""Leakage-aware demand features: all rolling statistics use shifted history."""
import pandas as pd

FEATURES = ["lag_1","lag_2","lag_3","rolling_3","month_num","trend","part_code","region_code"]

def make_features(df):
    df = df.copy().sort_values(["part","region","month"]).reset_index(drop=True)
    df["month"] = pd.to_datetime(df["month"])
    group = df.groupby(["part","region"], sort=False)["demand"]
    for k in (1,2,3):
        df[f"lag_{k}"] = group.shift(k)
    df["rolling_3"] = group.transform(lambda s: s.shift(1).rolling(3).mean())
    df["month_num"] = df.month.dt.month
    df["trend"] = (df.month.dt.year - df.month.dt.year.min()) * 12 + df.month.dt.month
    df["part_code"] = pd.Categorical(df.part, categories=sorted(df.part.unique())).codes
    df["region_code"] = pd.Categorical(df.region, categories=sorted(df.region.unique())).codes
    return df.dropna(subset=FEATURES).reset_index(drop=True)

def chronological_split(df):
    dates = sorted(df.month.unique())
    if len(dates) < 10:
        raise ValueError("At least 10 feature-ready months required")
    a, b = int(len(dates)*.6), int(len(dates)*.8)
    return (df[df.month.isin(dates[:a])].copy(), df[df.month.isin(dates[a:b])].copy(), df[df.month.isin(dates[b:])].copy())
