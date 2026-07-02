import os
import pandas as pd
import numpy as np
import json

from prophet import Prophet
from google.cloud import bigquery
from sklearn.metrics import mean_absolute_error, mean_squared_error

from google.oauth2 import service_account
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
# =========================
# 0. CONFIG
# =========================
PROJECT_ID = "damiu-nasqua-488213"

credentials_info = json.loads(
    os.getenv("GCP_CREDENTIALS")
)

credentials = service_account.Credentials.from_service_account_info(
    credentials_info
)

client = bigquery.Client(
    project=PROJECT_ID,
    credentials=credentials
)

# =========================
# 1. LOAD DATA
# =========================
query = """
SELECT
  w.tanggal AS ds,
  SUM(f.jumlah_galon) AS y
FROM `damiu-nasqua-488213.datamart_damiu_nasqua.fact_penjualan` f
JOIN `damiu-nasqua-488213.datamart_damiu_nasqua.dim_waktu` w
ON f.id_waktu = w.id_waktu
GROUP BY w.tanggal
ORDER BY w.tanggal
"""

df = client.query(query).to_dataframe()
if df.empty:
    raise ValueError("Query returned empty dataframe")

df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

# =========================
# 2. DAILY COMPLETE SERIES
# =========================
all_dates = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")

df_daily = (
    pd.DataFrame({"ds": all_dates})
    .merge(df, on="ds", how="left")
)

df_daily["y"] = df_daily["y"].fillna(0).astype(float)

# =========================
# 3. FEATURES
# =========================
df_daily["is_libur"] = (df_daily["y"] == 0).astype(int)
df_daily["is_buka"] = (df_daily["y"] > 0).astype(int)
df_daily["is_weekend"] = (df_daily["ds"].dt.dayofweek >= 5).astype(int)

# =========================
# 4. WEEKLY AGGREGATION
# =========================
df_weekly = (
    df_daily
    .resample("W", on="ds")
    .agg({
        "y": "sum",
        "is_libur": "sum",
        "is_buka": "sum",
        "is_weekend": "sum"
    })
    .reset_index()
)

# =========================
# 5. FEATURES ENGINEERING
# =========================
df_weekly["lag_1"] = df_weekly["y"].shift(1)

df_weekly = df_weekly.replace([np.inf, -np.inf], 0).fillna(0)

# =========================
# 6. TRAIN TEST SPLIT
# =========================
split = int(len(df_weekly) * 0.8)

train = df_weekly.iloc[:split].copy()
test = df_weekly.iloc[split:].copy()

train["y"] = np.log1p(train["y"])

regressors = [
    "is_libur", "is_buka", "is_weekend",
    "lag_1"
]

# =========================
# 7. MODEL
# =========================
model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=5,
    seasonality_mode="multiplicative"
)

for r in regressors:
    model.add_regressor(r)

model.fit(train[["ds", "y"] + regressors])

# =========================
# 8. FORECAST FULL HISTORICAL
# =========================
future = df_weekly[["ds"] + regressors]

forecast = model.predict(future)
forecast["yhat"] = np.expm1(forecast["yhat"]).clip(lower=0)

# =========================
# 9. EVALUATION
# =========================
df_eval = test[["ds", "y"]].copy().reset_index(drop=True)
df_eval["yhat"] = forecast["yhat"].tail(len(test)).values

mae = mean_absolute_error(df_eval["y"], df_eval["yhat"])
mape = np.mean(np.abs((df_eval['y'] - df_eval['yhat']) / (df_eval['y'] + 1))) * 100

print("\n===== EVALUATION =====")

print("MAE   :", round(mae, 2))
print("MAPE  :", round(mape, 2))

# =========================
# 10. RETRAIN FULL MODEL
# =========================
full_model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=5,
    seasonality_mode="multiplicative"
)

for r in regressors:
    full_model.add_regressor(r)

full_model.fit(df_weekly.assign(y=np.log1p(df_weekly["y"]))[["ds","y"]+regressors])

# =========================
# 11. FUTURE FORECAST
# =========================
future_weeks = 8

future_dates = pd.date_range(
    df_weekly["ds"].max() + pd.Timedelta(weeks=1),
    periods=future_weeks,
    freq="W"
)

future_df = pd.DataFrame({"ds": future_dates})

future_df["is_libur"] = 0
future_df["is_buka"] = 7
future_df["is_weekend"] = 2

last = df_weekly["y"].iloc[-1]

future_df["lag_1"] = last

future_all = pd.concat([
    df_weekly[["ds"] + regressors],
    future_df[["ds"] + regressors]
])
future_all = future_all.reset_index(drop=True)

forecast_final = full_model.predict(future_all)


# =========================
# 12. EXPORT KE BIGQUERY
# =========================
target_table = f"{PROJECT_ID}.datamart_damiu_nasqua.forecast_result_weekly"

output = forecast_final[
    [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "trend"
    ]
].copy()

# inverse transform
output["yhat"] = np.expm1(output["yhat"]).clip(lower=0)
output["yhat_lower"] = np.expm1(output["yhat_lower"]).clip(lower=0)
output["yhat_upper"] = np.expm1(output["yhat_upper"]).clip(lower=0)

# gabungkan actual historical
output = output.merge(
    df_weekly[["ds", "y"]],
    on="ds",
    how="left"
)

# rename columns
output = output.rename(columns={
    "ds": "minggu",
    "y": "actual",
    "yhat": "forecast",
    "yhat_lower": "forecast_lower",
    "yhat_upper": "forecast_upper"
})

# rounding
numeric_cols = [
    "actual",
    "forecast",
    "forecast_lower",
    "forecast_upper",
    "trend"
]

output[numeric_cols] = output[numeric_cols].round(2)

job = client.load_table_from_dataframe(
    output,
    target_table,
    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
)

job.result()

print("SUCCESS → BigQuery updated:", target_table)