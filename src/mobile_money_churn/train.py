"""Train a baseline and a boosted model, then look at them the way a
retention team would: if we can only call the top 10% riskiest customers,
how many of the real churners do we catch?

    uv run python -m mobile_money_churn.train
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, TARGET, build_features
from .simulate import simulate_customers

REPORTS = Path("reports")


def capture_at(y_true, scores, frac=0.10):
    """Share of all churners that land in the top `frac` of scores."""
    k = int(len(scores) * frac)
    top = np.argsort(-scores)[:k]
    return y_true[top].sum() / y_true.sum()


def make_models(numeric_cols):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    logreg = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))

    pre_gb = ColumnTransformer(
        [
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    gbm = make_pipeline(
        pre_gb,
        HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=0,
        ),
    )
    return {"logistic_regression": logreg, "gradient_boosting": gbm}


def main():
    df = simulate_customers()
    X = build_features(df)
    y = df[TARGET].to_numpy().astype(int)
    print(f"{len(df):,} customers, churn rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=7
    )
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]

    results, scores = {}, {}
    for name, model in make_models(numeric_cols).items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        scores[name] = p
        results[name] = {
            "roc_auc": round(roc_auc_score(y_test, p), 4),
            "pr_auc": round(average_precision_score(y_test, p), 4),
            "capture_top10pct": round(float(capture_at(y_test, p, 0.10)), 4),
            "capture_top20pct": round(float(capture_at(y_test, p, 0.20)), 4),
        }
        print(name, results[name])

    REPORTS.joinpath("figures").mkdir(parents=True, exist_ok=True)
    REPORTS.joinpath("metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    # precision-recall
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, p in scores.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, label=name.replace("_", " "))
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label="random")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision vs recall on held-out customers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/precision_recall.png", dpi=120)

    # cumulative gains - this is the chart the business side actually reads
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fracs = np.linspace(0, 1, 101)
    for name, p in scores.items():
        ax.plot(fracs * 100, [capture_at(y_test, p, f) * 100 if f else 0 for f in fracs],
                label=name.replace("_", " "))
    ax.plot([0, 100], [0, 100], color="grey", ls="--", lw=1, label="random")
    ax.set_xlabel("% of customers contacted (riskiest first)")
    ax.set_ylabel("% of churners reached")
    ax.set_title("Cumulative gains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/cumulative_gains.png", dpi=120)

    # which features carry the boosted model
    gbm = make_models(numeric_cols)["gradient_boosting"].fit(X_train, y_train)
    imp = permutation_importance(gbm, X_test, y_test, scoring="average_precision",
                                 n_repeats=5, random_state=0)
    order = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    order.plot.barh(ax=ax, color="#3b7dd8")
    ax.set_xlabel("drop in PR-AUC when shuffled")
    ax.set_title("Top 10 features (permutation importance)")
    fig.tight_layout()
    fig.savefig(REPORTS / "figures/feature_importance.png", dpi=120)
    print("saved reports/metrics.json and figures")


if __name__ == "__main__":
    main()
