"""Constrained one-period inventory recommendations; synthetic planning assumptions."""
from pathlib import Path
import pandas as pd
from ortools.linear_solver import pywraplp

def recommend(frame, budget=18000, capacity=4500, service_target=.9):
    if not 0 <= service_target <= 1: raise ValueError("service_target must be within [0,1]")
    if budget < 0 or capacity < 0: raise ValueError("budget and capacity must be nonnegative")
    df = frame.copy().reset_index(drop=True)
    required = {"part","region","prediction","unit_cost","unit_volume"}
    if not required.issubset(df): raise ValueError(f"Missing columns: {required-set(df)}")
    if (df[["prediction","unit_cost","unit_volume"]] < 0).any().any(): raise ValueError("negative input")
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None: raise RuntimeError("CBC unavailable")
    quantities=[]; shortages=[]
    for i,row in df.iterrows():
        demand=max(0,round(float(row.prediction)))
        q=solver.IntVar(0, max(1,2*demand), f"q_{i}")
        s=solver.NumVar(0,solver.infinity(),f"short_{i}")
        solver.Add(q+s >= demand)
        solver.Add(q >= int(__import__("math").ceil(service_target*demand)))
        quantities.append(q); shortages.append(s)
    solver.Add(sum(quantities[i]*float(df.iloc[i].unit_cost) for i in range(len(df))) <= budget)
    solver.Add(sum(quantities[i]*float(df.iloc[i].unit_volume) for i in range(len(df))) <= capacity)
    solver.Minimize(sum(quantities[i]*float(df.iloc[i].unit_cost) + shortages[i]*float(df.iloc[i].unit_cost)*4 for i in range(len(df))))
    status=solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise ValueError("No feasible optimum for budget, capacity and service targets")
    df["recommended_order"]=[int(round(q.solution_value())) for q in quantities]
    df["estimated_shortage"]=[round(s.solution_value(),2) for s in shortages]
    return df

def run():
    path=Path("artifacts/forecasts.csv")
    if not path.exists(): raise FileNotFoundError("Run python -m supplyguard.train first")
    # Optimize the latest held-out month only, not all historical months together.
    frame=pd.read_csv(path)
    frame=frame[frame.month==frame.month.max()]
    out=recommend(frame)
    out.to_csv("artifacts/recommendations.csv",index=False)
    return out

if __name__=="__main__":
    print(run().to_string(index=False))
