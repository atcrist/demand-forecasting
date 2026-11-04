"""
Monthly demand forecasting by store x item: statistical vs. machine-learning baselines.

Pipeline
  1. Aggregate daily sales to monthly (store, item, month).
  2. Hold out the last 12 months (2017) as the test window.
  3. Fit four baselines on the training window:
       - Seasonal naive (same month last year)              -> reference
       - ETS (Holt-Winters, additive trend + seasonality)    -> per series
       - SARIMA (1,1,1)(1,1,0)12                             -> per series
       - LightGBM global model with lag / rolling / calendar features
         (recursive multi-step forecast)
  4. Score every model with demand-planning metrics:
       WAPE, MAPE, bias (signed % error), at 1-, 3-, 6- and 12-month horizons,
       plus Forecast Value Added (FVA) versus the seasonal-naive reference.
  5. Segment the portfolio by volume and variability (coefficient of variation)
     and recommend: automate / plan by exception / planner judgement.

Run:  python forecast.py
Outputs land in ./outputs/
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA = Path("data/train.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

TEST_MONTHS = 12          # hold-out window
HORIZONS = [1, 3, 6, 12]  # months ahead at which we report accuracy
SEASON = 12

# --------------------------------------------------------------------------- #
# 1. Load and aggregate
# --------------------------------------------------------------------------- #
daily = pd.read_csv(DATA, parse_dates=["date"])
daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
monthly = (daily.groupby(["store", "item", "month"], as_index=False)["sales"].sum()
                .sort_values(["store", "item", "month"]))
monthly["series_id"] = monthly["store"].astype(str) + "-" + monthly["item"].astype(str)

months = sorted(monthly["month"].unique())
train_months, test_months = months[:-TEST_MONTHS], months[-TEST_MONTHS:]
print(f"{monthly['series_id'].nunique()} series, {len(months)} months "
      f"({months[0]:%Y-%m} to {months[-1]:%Y-%m}); test window {test_months[0]:%Y-%m} to {test_months[-1]:%Y-%m}")

wide = monthly.pivot(index="month", columns="series_id", values="sales")
train_wide, test_wide = wide.loc[train_months], wide.loc[test_months]

# --------------------------------------------------------------------------- #
# 2. Statistical baselines (per series)
# --------------------------------------------------------------------------- #
def seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    last_year = y.values[-SEASON:]
    return np.tile(last_year, int(np.ceil(h / SEASON)))[:h]


def ets(y: pd.Series, h: int) -> np.ndarray:
    model = ExponentialSmoothing(y, trend="add", seasonal="add",
                                 seasonal_periods=SEASON, initialization_method="estimated")
    return model.fit(optimized=True).forecast(h).values


def sarima(y: pd.Series, h: int) -> np.ndarray:
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 0, SEASON),
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False).forecast(h).values


forecasts = {}
for name, fn in [("seasonal_naive", seasonal_naive), ("ets", ets), ("sarima", sarima)]:
    print(f"fitting {name} on {train_wide.shape[1]} series ...")
    preds = {sid: fn(train_wide[sid], TEST_MONTHS) for sid in train_wide.columns}
    forecasts[name] = pd.DataFrame(preds, index=test_months)

# --------------------------------------------------------------------------- #
# 3. LightGBM global model (one model for all series, recursive forecasting)
# --------------------------------------------------------------------------- #
LAGS = [1, 2, 3, 6, 12]
ROLL = [3, 6, 12]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["series_id", "month"]).copy()
    g = df.groupby("series_id")["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLL:
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
    df["month_of_year"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    df["store"] = df["store"].astype("category")
    df["item"] = df["item"].astype("category")
    return df


FEATURES = [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLL] + \
           ["month_of_year", "year", "store", "item"]

train_long = monthly[monthly["month"].isin(train_months)].copy()
feat = make_features(train_long).dropna(subset=FEATURES)
params = dict(objective="regression", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, verbose=-1, seed=42)
print("training LightGBM ...")
booster = lgb.train(params, lgb.Dataset(feat[FEATURES], feat["sales"],
                                        categorical_feature=["store", "item"]),
                    num_boost_round=600)

# recursive forecast: predict month t, append it, build features for t+1
history = train_long.copy()
lgb_preds = []
for m in test_months:
    nxt = history[["store", "item", "series_id"]].drop_duplicates().copy()
    nxt["month"], nxt["sales"] = m, np.nan
    tmp = make_features(pd.concat([history, nxt], ignore_index=True))
    row = tmp[tmp["month"] == m]
    yhat = booster.predict(row[FEATURES])
    row = row.assign(sales=np.clip(yhat, 0, None))
    lgb_preds.append(row[["series_id", "month", "sales"]])
    history = pd.concat([history, row[["store", "item", "series_id", "month", "sales"]]],
                        ignore_index=True)
forecasts["lightgbm"] = (pd.concat(lgb_preds).pivot(index="month", columns="series_id",
                                                     values="sales").loc[test_months])

# --------------------------------------------------------------------------- #
# 4. Demand-planning metrics
# --------------------------------------------------------------------------- #
def wape(actual, pred):
    return np.abs(actual - pred).sum() / np.abs(actual).sum() * 100


def mape(actual, pred):
    mask = actual != 0
    return (np.abs(actual - pred)[mask] / np.abs(actual)[mask]).mean() * 100


def bias(actual, pred):
    """Signed error as % of actual demand: + = over-forecast, - = under-forecast."""
    return (pred - actual).sum() / actual.sum() * 100


rows = []
for name, fc in forecasts.items():
    for h in HORIZONS:
        a, p = test_wide.iloc[:h].values.ravel(), fc.iloc[:h].values.ravel()
        rows.append(dict(model=name, horizon_months=h,
                         WAPE=wape(a, p), MAPE=mape(a, p), bias=bias(a, p)))
scores = pd.DataFrame(rows)
ref = scores[scores["model"] == "seasonal_naive"].set_index("horizon_months")["WAPE"]
scores["FVA_vs_naive_pts"] = scores.apply(
    lambda r: ref[r["horizon_months"]] - r["WAPE"], axis=1)   # + = model adds value
scores = scores.round(2)
scores.to_csv(OUT / "model_scores.csv", index=False)
print("\nAccuracy by model and horizon (test window):")
print(scores.pivot(index="horizon_months", columns="model", values="WAPE").round(2))

# --------------------------------------------------------------------------- #
# 5. Portfolio segmentation and planning recommendation
# --------------------------------------------------------------------------- #
per_series = []
for sid in test_wide.columns:
    a = test_wide[sid].values
    stats = dict(series_id=sid,
                 avg_monthly_demand=train_wide[sid].mean(),
                 cv=train_wide[sid].std() / train_wide[sid].mean())
    for name, fc in forecasts.items():
        stats[f"wape_{name}"] = wape(a, fc[sid].values)
    per_series.append(stats)
per_series = pd.DataFrame(per_series)

model_cols = [c for c in per_series.columns if c.startswith("wape_")]
per_series["best_model"] = per_series[model_cols].idxmin(axis=1).str.replace("wape_", "")
per_series["best_wape"] = per_series[model_cols].min(axis=1)

vol_cut = per_series["avg_monthly_demand"].median()
cv_cut = per_series["cv"].quantile(0.75)


def recommend(r):
    if r["best_wape"] <= 10:
        return "Automate (statistical baseline)"
    if r["best_wape"] <= 20 or r["cv"] <= cv_cut:
        return "Plan by exception"
    return "Planner judgement"


per_series["volume_segment"] = np.where(per_series["avg_monthly_demand"] >= vol_cut, "High", "Low")
per_series["variability_segment"] = np.where(per_series["cv"] > cv_cut, "Erratic", "Stable")
per_series["recommendation"] = per_series.apply(recommend, axis=1)
per_series.round(3).to_csv(OUT / "per_series_results.csv", index=False)

summary = (per_series.groupby("recommendation")
           .agg(series=("series_id", "count"),
                share_of_demand_pct=("avg_monthly_demand",
                                     lambda s: s.sum() / per_series["avg_monthly_demand"].sum() * 100),
                median_best_wape=("best_wape", "median"))
           .round(1))
summary.to_csv(OUT / "planning_recommendation.csv")
print("\nPlanning recommendation by segment:")
print(summary)
print("\nBest model share:")
print(per_series["best_model"].value_counts())

# --------------------------------------------------------------------------- #
# 6. Charts
# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))
for name in forecasts:
    s = scores[scores["model"] == name]
    ax.plot(s["horizon_months"], s["WAPE"], marker="o", label=name)
ax.set_xlabel("Forecast horizon (months)"); ax.set_ylabel("WAPE (%)")
ax.set_title("Forecast accuracy by horizon — test window 2017"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "wape_by_horizon.png", dpi=150)

example = per_series.sort_values("avg_monthly_demand", ascending=False)["series_id"].iloc[0]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(wide.index, wide[example], color="black", label="actual")
for name, fc in forecasts.items():
    ax.plot(fc.index, fc[example], linestyle="--", label=name)
ax.axvline(test_months[0], color="grey", alpha=.5)
ax.set_title(f"Series {example} (store-item): actual vs. forecasts"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "example_series.png", dpi=150)
print(f"\nDone. Results in {OUT}/")
