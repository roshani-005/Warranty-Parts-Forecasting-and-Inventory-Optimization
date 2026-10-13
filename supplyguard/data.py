"""Deterministic synthetic data; no real customer records."""
from pathlib import Path
import numpy as np
import pandas as pd

PARTS = ["battery", "display", "motor", "sensor", "controller", "valve"]
REGIONS = ["north", "south", "east", "west"]

def generate(seed=42, months=36):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=months, freq="MS")
    rows = []
    for p, part in enumerate(PARTS):
        for r, region in enumerate(REGIONS):
            for t, date in enumerate(dates):
                seasonal = 1 + .20 * np.sin(2 * np.pi * (date.month - 1) / 12)
                base = (18 + 4 * p + 3 * r) * (1 + .006 * t) * seasonal
                demand = max(0, int(round(base + rng.normal(0, 4))))
                lead = max(1, int(round(5 + p % 3 + r % 2 + rng.normal(0, 2))))
                promised = 7 + p % 3
                rows.append((date, part, region, demand, lead, promised, int(lead > promised), 20 + 8*p, 2 + p))
    return pd.DataFrame(rows, columns=["month","part","region","demand","lead_days","promised_days","delayed","unit_cost","unit_volume"])

def save(path="data/synthetic_warranty.csv"):
    df = generate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

if __name__ == "__main__":
    print(f"Generated {len(save())} synthetic observations")
