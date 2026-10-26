import numpy as np
import pandas as pd

TXN_COLS = [f"txn_m{m}" for m in range(1, 7)]
CATEGORICAL = ["region"]
TARGET = "churned_next_60d"


def activity_slope(txns: np.ndarray) -> np.ndarray:
    """Least squares slope of log activity across the six months."""
    y = np.log1p(txns)
    x = np.arange(y.shape[1]) - (y.shape[1] - 1) / 2
    return (y * x).sum(axis=1) / (x**2).sum()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    txns = df[TXN_COLS].to_numpy()
    out = df.drop(columns=["customer_id", TARGET], errors="ignore").copy()

    out["txn_total"] = txns.sum(axis=1)
    out["txn_slope"] = activity_slope(txns)
    # last two months vs the first four - the single most useful signal
    recent = txns[:, -2:].mean(axis=1)
    earlier = txns[:, :4].mean(axis=1)
    out["recent_vs_earlier"] = (recent + 1) / (earlier + 1)
    out["zero_months"] = (txns == 0).sum(axis=1)
    out["agent_nearby"] = out["agent_nearby"].astype(int)
    return out
