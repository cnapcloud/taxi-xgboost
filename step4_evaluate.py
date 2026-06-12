"""
NYC Taxi MLOps Pipeline - Step 4: Model Evaluation (MLflow Registry 기반)
========================================================================
목적:
    fare(요금) 및 ETA(소요시간) 두 모델 각각에 대해 MLflow Model Registry의
    챔피언(Staging 모델)과 챌린저(Latest 등록 버전)를 가져와 성능(RMSE)을 대조 비교한다.
    통과 시 챌린저를 Staging으로 승격하고, 버전 정보를 리포트에 포함한다.

핵심 검증 및 예외 게이트웨이 규칙:
    1. 개선도 검증: 챌린저가 챔피언 대비 설정한 개선 기준치(--threshold) 이상 성능이 향상되면 승인한다.
       (기존 모델 대비 1% 이상 개선 조건: --threshold 0.01)
    2. 최초 등록 예외: Registry에 기존 Staging(챔피언) 모델이 없거나, 챌린저가 'Version 1'인 경우
                     최초 등록 시점으로 판단하여 검증을 자동 통과(Auto-Promote)시킨다.
    3. 동일 모델 예외: 챔피언과 챌린저의 RMSE 성능이 소수점까지 완전히 일치(동일 버전 재평가 등)하는 경우,
                     파이프라인 데드락을 방지하기 위해 경고 로그 출력 후 자동 승인한다.
    4. 파이프라인 차단: 두 모델 중 하나라도 위의 승인 조건을 만족하지 못하면 파이프라인을 즉시 중단(Halted)한다.

변경 사항:
    - 평가 통과 시 챌린저를 Staging으로 승격 (transition_model_version_stage)
    - 리포트에 challenger_version 추가 → step5가 별도 등록 없이 바로 Production 승격 가능
    - 동일 모델 예외 처리 추가 (RMSE 완전 일치 시 자동 승인)

실행 예시(최초 프로모션):
    python step4_evaluate.py \
        --mlflow-uri http://mlflow.cnapcloud.com \
        --test-data data/raw/ \
        --output reports/evaluation/ \
        --threshold 0.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "pickup_hour", "pickup_dayofweek", "pickup_month",
    "trip_distance", "passenger_count",
    "PULocationID", "DOLocationID",
    "is_weekend", "is_rush_hour",
]

# 평가할 모델 목록 (타겟 컬럼, MLflow Registry에 등록된 모델 이름, 단위)
MODEL_CONFIGS = [
    ("fare_amount",       "nyc-taxi-fare", "$"),
    ("trip_duration_min", "nyc-taxi-eta",  "min"),
]


# ──────────────────────────────────────────────
# 1. 피처 엔지니어링
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"]      = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"]     = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"]       = (df["pickup_dayofweek"] >= 5).astype(int)
    df["is_rush_hour"]     = df["pickup_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    df = df[
        (df["fare_amount"] > 0)       & (df["fare_amount"] < 300) &
        (df["trip_distance"] > 0)     & (df["trip_distance"] < 100) &
        (df["trip_duration_min"] > 0) & (df["trip_duration_min"] < 180) &
        (df["passenger_count"] >= 1)  & (df["passenger_count"] <= 6)
    ]
    return df[FEATURE_COLS + ["fare_amount", "trip_duration_min"]].dropna()


# ──────────────────────────────────────────────
# 2. MLflow 모델 로드
# ──────────────────────────────────────────────
def load_champion_and_challenger(mlflow_uri: str, model_name: str):
    """MLflow Registry에서 챔피언(Staging)과 챌린저(Latest) 모델을 쌍으로 로드"""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    # [1] 챌린저 모델 로드 (가장 최근에 등록된 버전)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No registered model found: '{model_name}'")

    latest_version = max(versions, key=lambda v: int(v.version))
    log.info(f"[{model_name}] Loading Challenger (Latest Version: {latest_version.version})...")
    challenger_model = mlflow.xgboost.load_model(f"models:/{model_name}/{latest_version.version}")

    # [2] 챔피언 모델 로드 (현재 Staging 라벨을 가진 모델)
    champion_model = None
    try:
        log.info(f"[{model_name}] Attempting to load Champion (staging)...")
        champion_model = mlflow.xgboost.load_model(f"models:/{model_name}@staging")
    except Exception as e:
        log.warning(f"[{model_name}] No Staging model found. Challenger will be auto-promoted. (Reason: {e})")

    return champion_model, challenger_model, latest_version, client


def run_prediction_for_eval(model, input_df: pd.DataFrame) -> np.ndarray:
    """Booster와 Regressor 객체 유형을 자동 판별하여 고속 추론 수행"""
    input_df_processed = input_df.astype(float)
    if isinstance(model, xgb.Booster):
        dtrain = xgb.DMatrix(input_df_processed, nthread=-1)
        return model.predict(dtrain)
    else:
        return model.predict(input_df_processed)


def compute_metrics(y_true, y_pred, unit: str) -> dict:
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.abs((y_true.values - y_pred) / np.clip(y_true.values, 0.01, None)).mean() * 100)
    p90  = float(np.percentile(np.abs(y_true.values - y_pred), 90))
    return {
        "rmse": round(rmse, 4), "mae": round(mae, 4),
        "r2": round(r2, 4), "mape_pct": round(mape, 2),
        "p90_abs_error": round(p90, 4), "unit": unit,
    }


# ──────────────────────────────────────────────
# 3. Staging 승격  ← 추가
# ──────────────────────────────────────────────
def promote_to_staging(client, model_name, version):
    # 기존 champion alias 제거
    try:
        client.delete_registered_model_alias(model_name, "staging")
    except Exception:
        pass

    # 새 버전에 champion alias 설정
    client.set_registered_model_alias(model_name, "staging", version)
    

# ──────────────────────────────────────────────
# 4. 메인 파이프라인 제어 흐름
# ──────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 테스트 배치 데이터 로드
    p = Path(args.test_data)
    files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    if not files:
        log.error(f"No test files found at path: {args.test_data}")
        sys.exit(1)

    df = engineer_features(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    log.info("Loaded evaluation test set size: %d rows", len(df))

    X_features = df[FEATURE_COLS]
    y_dict = {
        "fare_amount":      df["fare_amount"],
        "trip_duration_min": df["trip_duration_min"],
    }

    all_results = []
    rejected = []

    for target_col, model_name, unit in MODEL_CONFIGS:
        print(f"\n--- Evaluating [{model_name}] ---")
        y_true = y_dict[target_col]

        champion, challenger, latest_version, client = load_champion_and_challenger(
            args.mlflow_uri, model_name
        )

        # [1] 챌린저 평가
        pred_c = run_prediction_for_eval(challenger, X_features)
        challenger_metrics = compute_metrics(y_true, pred_c, unit)
        log.info("[%s] Challenger – RMSE=%.4f%s | MAE=%.4f%s | R²=%.4f",
                 target_col, challenger_metrics["rmse"], unit,
                 challenger_metrics["mae"], unit, challenger_metrics["r2"])

        # [2] 챔피언 평가 및 대조 비교
        if champion is not None:
            pred_ch = run_prediction_for_eval(champion, X_features)
            champion_metrics = compute_metrics(y_true, pred_ch, unit)
            log.info("[%s] Champion   – RMSE=%.4f%s | MAE=%.4f%s | R²=%.4f",
                     target_col, champion_metrics["rmse"], unit,
                     champion_metrics["mae"], unit, champion_metrics["r2"])

            # 동일 모델 예외 처리 ← 추가
            if challenger_metrics["rmse"] == champion_metrics["rmse"]:
                log.warning(f"[{model_name}] RMSE identical. Auto-promoting to prevent deadlock.")
                promote = True
                decision = "PROMOTE (identical RMSE)"
            else:
                improvement = (champion_metrics["rmse"] - challenger_metrics["rmse"]) / champion_metrics["rmse"]
                promote = improvement >= args.threshold
                decision = "PROMOTE" if promote else "KEEP_CHAMPION"

            comparison = {
                "champion_rmse":      champion_metrics["rmse"],
                "challenger_rmse":    challenger_metrics["rmse"],
                "improvement_pct":    round(improvement * 100, 3) if champion_metrics["rmse"] != challenger_metrics["rmse"] else 0.0,
                "threshold_pct":      round(args.threshold * 100, 3),
                "promote_challenger": promote,
                "decision":           decision,
            }
        else:
            # 최초 등록 예외
            champion_metrics = {}
            comparison = {
                "champion_rmse":      None,
                "challenger_rmse":    challenger_metrics["rmse"],
                "improvement_pct":    None,
                "promote_challenger": True,
                "decision":           "PROMOTE (no champion)",
            }

        # [3] 통과 시 Staging 승격 ← 추가
        if comparison["promote_challenger"]:
            promote_to_staging(client, model_name, str(latest_version.version))

        result = {
            "target":              target_col,
            "challenger_version":  str(latest_version.version),  # ← 추가: step5에서 사용
            "unit":                unit,
            "challenger_metrics":  challenger_metrics,
            "champion_metrics":    champion_metrics,
            "comparison":          comparison,
        }
        all_results.append(result)

        if not comparison["promote_challenger"]:
            rejected.append(target_col)

    # 평가 통합 리포트 JSON 저장
    report = {
        "models":           all_results,
        "rejected_targets": rejected,
        "overall_promote":  len(rejected) == 0,
    }
    report_path = out / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Evaluation report saved: %s", report_path)

    # 결과 요약 CLI 출력
    print("\n" + "=" * 60)
    print("  MLFLOW MODEL EVALUATION SUMMARY")
    print("=" * 60)
    for r in all_results:
        label = "요금 예측 (Fare)" if "fare" in r["target"] else "ETA 예측"
        cmp   = r["comparison"]
        print(f"\n  [{label}]")
        if cmp.get("champion_rmse"):
            print(f"    Champion (Staging) RMSE : {cmp['champion_rmse']:.4f} {r['unit']}")
        print(f"    Challenger (Latest) RMSE: {cmp['challenger_rmse']:.4f} {r['unit']}")
        if cmp.get("improvement_pct") is not None:
            print(f"    Improvement             : {cmp['improvement_pct']:.2f}%")
        print(f"    Decision                : {'' if cmp['promote_challenger'] else ''}{cmp['decision']}")
        if cmp["promote_challenger"]:
            print(f"    Staged Version          : {r['challenger_version']}")
    print("\n" + "=" * 60)

    # 파이프라인 중단 게이트웨이
    if rejected:
        log.warning("Rejected targets: %s. Pipeline HALTED.", rejected)
        sys.exit(2)

    log.info("Step 4 complete. Both models promoted to Staging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi – Step 4: Evaluate + Staging Promote")
    parser.add_argument("--mlflow-uri",  default="http://mlflow.cnapcloud.com", help="MLflow Tracking/Registry URI")
    parser.add_argument("--test-data",   default="data/raw/",                   help="Test parquet path")
    parser.add_argument("--output",      default="reports/evaluation/",         help="Output directory for reports")
    parser.add_argument("--threshold",   type=float, default=0.01,              help="Min RMSE improvement ratio (e.g. 0.01 = 1%)")
    args = parser.parse_args()
    main(args)