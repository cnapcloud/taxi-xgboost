"""
NYC Taxi MLOps Pipeline - Step 5: Model Registration (등록)
============================================================
목적: fare(요금) / ETA 두 모델을 각각 MLflow Model Registry의
      staging alias에서 로드하여 smoke test 후 production alias로 전환한다.

변경 사항:
    - register_one() 제거: step3에서 이미 등록, step4에서 staging alias 부여 완료
    - evaluation-report 제거: staging alias로 버전 직접 조회
    - deprecated transition_model_version_stage() → set_registered_model_alias() 로 변경
    - staging alias 로 모델 로드 후 smoke test 통과 시 production alias 전환

실행:
    python step5_register.py \
        --mlflow-uri http://mlflow.cnapcloud.com \
        --auto-promote
"""

import argparse
import logging

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_REGISTRY_CONFIGS = [
    ("fare_amount",       "nyc-taxi-fare"),
    ("trip_duration_min", "nyc-taxi-eta"),
]

DUMMY_INPUT = {
    "pickup_hour": 14, "pickup_dayofweek": 2, "pickup_month": 6,
    "trip_distance": 3.5, "passenger_count": 1,
    "PULocationID": 161, "DOLocationID": 237,
    "is_weekend": 0, "is_rush_hour": 0,
}


# ──────────────────────────────────────────────
# 1. staging alias로 버전 조회
# ──────────────────────────────────────────────
def get_staging_version(client: MlflowClient, model_name: str) -> str:
    """staging alias가 가리키는 버전 번호 반환"""
    mv = client.get_model_version_by_alias(model_name, "staging")
    return mv.version


# ──────────────────────────────────────────────
# 2. Smoke test (staging alias 기반)
# ──────────────────────────────────────────────
def smoke_test(registrations: list[dict]) -> bool:
    import xgboost as xgb
    dummy = pd.DataFrame([DUMMY_INPUT])
    all_ok = True

    for reg in registrations:
        model_uri = f"models:/{reg['model_name']}@staging"
        model = mlflow.xgboost.load_model(model_uri)

        dmatrix = xgb.DMatrix(dummy)
        pred = model.predict(dmatrix)[0]
    
        unit = "$" if "fare" in reg["target"] else "min"
        log.info("Smoke test [%s]: %.2f %s", reg["target"], pred, unit)
        if pred <= 0:
            log.error("Smoke test FAILED for %s: prediction=%.4f", reg["target"], pred)
            all_ok = False

    return all_ok


# ──────────────────────────────────────────────
# 3. Production 전환 (alias 기반)
# ──────────────────────────────────────────────
def promote_to_production(client: MlflowClient, model_name: str, version: str) -> None:
    # 기존 production alias 제거
    try:
        client.delete_registered_model_alias(model_name, "production")
        log.info("[%s] Removed old production alias", model_name)
    except Exception:
        pass

    # production alias 부여
    client.set_registered_model_alias(model_name, "production", version)
    client.update_model_version(
        model_name, version,
        description="Auto-promoted to production by MLOps pipeline."
    )
    log.info("[%s] Version %s → production alias", model_name, version)


# ──────────────────────────────────────────────
# 4. 메인
# ──────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient()

    # staging alias로 버전 직접 조회
    registrations = []
    for target_col, model_name in MODEL_REGISTRY_CONFIGS:
        try:
            version = get_staging_version(client, model_name)
            log.info("[%s] staging alias → version %s", model_name, version)
        except Exception as e:
            raise RuntimeError(f"[{model_name}] staging alias not found. step4가 완료되었는지 확인하세요. ({e})")

        registrations.append({
            "target":     target_col,
            "model_name": model_name,
            "version":    version,
        })

    # Smoke test
    ok = smoke_test(registrations)
    if not ok:
        raise RuntimeError("Smoke test failed. Manual rollback required.")
    log.info("All smoke tests passed.")

    # Production 전환
    if args.auto_promote:
        for reg in registrations:
            promote_to_production(client, reg["model_name"], reg["version"])

    # 최종 요약
    print("\n" + "=" * 60)
    print("  MODEL REGISTRATION SUMMARY")
    print("=" * 60)
    for reg in registrations:
        label = "요금 예측 (Fare)" if "fare" in reg["target"] else "ETA 예측"
        print(f"\n  [{label}]")
        print(f"    Model   : {reg['model_name']}")
        print(f"    Version : {reg['version']}")
        print(f"    Alias   : {'production' if args.auto_promote else 'staging'}")
    print("\n" + "=" * 60)

    log.info("Step 5 complete. %d models promoted to production.", len(registrations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi - Step 5: Promote to Production")
    parser.add_argument("--mlflow-uri",   default="http://mlflow.cnapcloud.com", help="MLflow tracking URI")
    parser.add_argument("--auto-promote", action="store_true",              help="Auto-promote to production alias")
    args = parser.parse_args()
    main(args)