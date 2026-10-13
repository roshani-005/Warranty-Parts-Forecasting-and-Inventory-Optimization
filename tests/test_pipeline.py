import pytest
from supplyguard.data import generate
from supplyguard.features import make_features, chronological_split
from supplyguard.train import wape
from supplyguard.optimize import recommend

def test_determinism():
    assert generate().equals(generate())

def test_no_future_leakage():
    df=make_features(generate())
    row=df.iloc[0]
    original=generate()
    history=original[(original.part==row.part)&(original.region==row.region)&(original.month<row.month)]
    assert row.lag_1==history.iloc[-1].demand
    assert row.rolling_3==history.tail(3).demand.mean()

def test_chronology():
    a,b,c=chronological_split(make_features(generate()))
    assert a.month.max()<b.month.min()<c.month.min()

def test_wape():
    assert wape([10,20],[10,15])==pytest.approx(5/30)

def test_optimizer():
    import pandas as pd
    df=pd.DataFrame([{"part":"a","region":"x","prediction":10,"unit_cost":2,"unit_volume":1}])
    result=recommend(df,budget=100,capacity=20)
    assert result.recommended_order.iloc[0]>=9
    with pytest.raises(ValueError): recommend(df,budget=1,capacity=1)
