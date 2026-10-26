import numpy as np

from mobile_money_churn.features import activity_slope, build_features
from mobile_money_churn.simulate import simulate_customers


def test_simulation_is_reproducible():
    a = simulate_customers(n=500, seed=1)
    b = simulate_customers(n=500, seed=1)
    assert a.equals(b)


def test_churn_rate_is_plausible():
    df = simulate_customers(n=5000)
    assert 0.05 < df["churned_next_60d"].mean() < 0.40


def test_slope_sign():
    rising = np.array([[1, 2, 4, 8, 16, 32]])
    falling = rising[:, ::-1]
    assert activity_slope(rising)[0] > 0
    assert activity_slope(falling)[0] < 0


def test_features_drop_id_and_target():
    X = build_features(simulate_customers(n=100))
    assert "customer_id" not in X
    assert "churned_next_60d" not in X
    assert X.isna().sum().sum() == 0
