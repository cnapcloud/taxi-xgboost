"""
NYC Taxi MLOps Pipeline - Step 1: Data Analysis (분석)
======================================================
목적: 원시 데이터의 통계 분포, 피처 상관관계, 시계열 패턴을 분석하고
      리포트(JSON + 시각화)를 생성한다.

실행:
    pip install pandas pyarrow matplotlib seaborn
    python step1_analyze.py --input data/raw/ --output reports/
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "PULocationID",
    "DOLocationID",
]

SAMPLE_PARQUET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
)


# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────
def load_data(input_path: str) -> pd.DataFrame:
    """로컬 parquet 파일 또는 디렉터리를 로드한다."""
    p = Path(input_path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {p}")
        log.info("Loading %d parquet file(s) from %s", len(files), p)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        log.info("Loading single parquet file: %s", p)
        df = pd.read_parquet(p)

    log.info("Raw shape: %s", df.shape)
    return df


# ──────────────────────────────────────────────
# 2. 전처리 (분석용 최소 정제)
# ──────────────────────────────────────────────
def preprocess_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """분석에 필요한 컬럼만 선택하고 datetime 피처를 파생한다."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # 파생 피처
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek  # 0=Mon
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

    # 분석용 필터 (극단값 제거)
    mask = (
        (df["fare_amount"] > 0) &
        (df["fare_amount"] < 300) &
        (df["trip_distance"] > 0) &
        (df["trip_distance"] < 100) &
        (df["trip_duration_min"] > 0) &
        (df["trip_duration_min"] < 180) &
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6)
    )
    df_clean = df[mask].reset_index(drop=True)
    log.info("After analysis filter: %d rows (%.1f%%)", len(df_clean), 100 * len(df_clean) / len(df))
    return df_clean


# ──────────────────────────────────────────────
# 3. 통계 리포트 생성
# ──────────────────────────────────────────────
def generate_statistics(df: pd.DataFrame) -> dict:
    """핵심 통계 지표를 딕셔너리로 반환한다."""
    numeric_cols = ["fare_amount", "trip_distance", "trip_duration_min", "passenger_count"]
    stats = df[numeric_cols].describe().round(4).to_dict()

    hourly = (
        df.groupby("pickup_hour")["fare_amount"]
        .agg(mean_fare="mean", trip_count="count")
        .round(2)
        .to_dict(orient="index")
    )

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = (
        df.groupby("pickup_dayofweek")["fare_amount"]
        .agg(mean_fare="mean", trip_count="count")
        .round(2)
        .rename(index=lambda x: dow_labels[x])
        .to_dict(orient="index")
    )

    report = {
        "total_records": len(df),
        "date_range": {
            "start": str(df["tpep_pickup_datetime"].min()),
            "end": str(df["tpep_pickup_datetime"].max()),
        },
        "descriptive_statistics": stats,
        "hourly_pattern": hourly,
        "day_of_week_pattern": daily,
    }
    return report


# ──────────────────────────────────────────────
# 4. 시각화
# ──────────────────────────────────────────────
def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NYC Taxi – Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # (1) 시간대별 평균 요금
    ax = axes[0, 0]
    hourly = df.groupby("pickup_hour")["fare_amount"].mean()
    ax.bar(hourly.index, hourly.values, color="steelblue", edgecolor="white")
    ax.set_title("Average Fare by Hour of Day")
    ax.set_xlabel("Hour (0–23)")
    ax.set_ylabel("Avg Fare ($)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))

    # (2) 요일별 탑승 건수
    ax = axes[0, 1]
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_counts = df.groupby("pickup_dayofweek").size()
    ax.bar(daily_counts.index, daily_counts.values, color="coral", edgecolor="white")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_labels)
    ax.set_title("Trip Count by Day of Week")
    ax.set_ylabel("Number of Trips")

    # (3) 요금 분포 (히스토그램)
    ax = axes[1, 0]
    sample = df["fare_amount"].sample(min(50_000, len(df)), random_state=42)
    ax.hist(sample, bins=60, color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.set_title("Fare Amount Distribution")
    ax.set_xlabel("Fare ($)")
    ax.set_ylabel("Frequency")

    # (4) 거리 vs 요금 산점도
    ax = axes[1, 1]
    sample_df = df.sample(min(10_000, len(df)), random_state=42)
    ax.scatter(
        sample_df["trip_distance"], sample_df["fare_amount"],
        alpha=0.2, s=8, color="teal"
    )
    ax.set_title("Trip Distance vs. Fare Amount")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Fare ($)")

    plt.tight_layout()
    out_path = output_dir / "eda_report.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Visualization saved: %s", out_path)


# ──────────────────────────────────────────────
# 5. 메인
# ──────────────────────────────────────────────
def main(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(input_path)
    df = preprocess_for_analysis(df_raw)

    stats = generate_statistics(df)
    report_path = out / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info("Analysis report saved: %s", report_path)

    create_visualizations(df, out)
    log.info("Step 1 (Analyze) complete. Outputs in %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi – Step 1: Analyze")
    parser.add_argument("--input", default="data/raw/", help="Parquet file or directory")
    parser.add_argument("--output", default="reports/", help="Output directory")
    args = parser.parse_args()
    main(args.input, args.output)
