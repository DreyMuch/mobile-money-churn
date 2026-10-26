"""Generate a fake customer base that behaves roughly like a mobile money wallet.

I can't publish real customer data, so this builds something with the same
shape: six months of activity per customer plus a churn flag for the next
60 days. The churn probability is driven by a handful of things I've seen
matter in practice (falling activity, failed transactions, no agent nearby,
short tenure) with plenty of noise on top so the model has to work for it.
"""

import numpy as np
import pandas as pd

REGIONS = ["Harare", "Bulawayo", "Mutare", "Gweru", "Masvingo", "Rural"]
REGION_WEIGHTS = [0.38, 0.17, 0.09, 0.08, 0.06, 0.22]
MONTHS = 6


def simulate_customers(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    region = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    rural = region == "Rural"
    tenure = rng.gamma(shape=2.0, scale=14.0, size=n).clip(1, 120).round()
    age = rng.normal(34, 10, size=n).clip(18, 75).round()
    agent_nearby = rng.random(n) < np.where(rural, 0.45, 0.85)

    # baseline monthly transaction rate, then a per-customer trend
    base_rate = rng.lognormal(mean=2.3, sigma=0.7, size=n)
    trend = rng.normal(0, 0.08, size=n)
    # a chunk of customers are already drifting away
    drifting = rng.random(n) < 0.18
    trend[drifting] -= rng.uniform(0.1, 0.35, size=drifting.sum())

    month_idx = np.arange(MONTHS)
    rates = base_rate[:, None] * np.exp(trend[:, None] * month_idx[None, :])
    txns = rng.poisson(rates)

    failed_rate = rng.beta(1.2, 30, size=n) + np.where(rural, 0.02, 0.0)
    p2p_share = rng.beta(4, 3, size=n)
    merchant_share = (1 - p2p_share) * rng.beta(2, 3, size=n)
    avg_balance = rng.lognormal(mean=3.0, sigma=1.1, size=n)
    support_calls = rng.poisson(0.3 + 4 * failed_rate)

    last_month = txns[:, -1]
    days_since_last = np.where(
        last_month > 0,
        rng.integers(0, 10, size=n),
        rng.integers(30, 90, size=n),
    )

    logit = (
        -1.6
        - 6.0 * trend
        + 8.0 * failed_rate
        - 0.012 * tenure
        - 0.6 * agent_nearby
        + 0.025 * days_since_last
        - 0.25 * np.log1p(avg_balance)
        + 0.15 * support_calls
        - 0.8 * merchant_share
        + rng.normal(0, 0.6, size=n)
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "region": region,
            "age": age.astype(int),
            "tenure_months": tenure.astype(int),
            "agent_nearby": agent_nearby,
            "avg_balance_usd": avg_balance.round(2),
            "p2p_share": p2p_share.round(3),
            "merchant_share": merchant_share.round(3),
            "failed_txn_rate": failed_rate.round(4),
            "support_calls_6m": support_calls,
            "days_since_last_txn": days_since_last,
        }
    )
    for m in range(MONTHS):
        df[f"txn_m{m + 1}"] = txns[:, m]
    df["churned_next_60d"] = churned
    return df


if __name__ == "__main__":
    from pathlib import Path

    out = Path("data/customers.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    simulate_customers().to_csv(out, index=False)
    print(f"wrote {out}")
