"""
NYC Taxi – 예측 사용 예시 (Fare + ETA)
=======================================
학습 완료 후 두 모델을 초기에 로드한 뒤, 루프 내에서는 순수 예측만 수행하는 스크립트.

실행:
    python predict_example.py
    python predict_example.py --mlflow-uri http://mlflow-svc:5000  # MLflow Registry 사용
"""

import argparse
from pathlib import Path
import pandas as pd
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# [1단계] 모델 로드 함수들 (초기 1회만 호출)
# ──────────────────────────────────────────────

def load_models_from_file(fare_model_path: str, eta_model_path: str):
    """로컬 파일(.json)에서 XGBoost 모델 로드"""
    if not Path(fare_model_path).exists() or not Path(eta_model_path).exists():
        print(f"\n  ⚠ 모델 파일 없음: {fare_model_path} 또는 {eta_model_path}")
        print("  Step 3 학습을 먼저 실행하세요: python step3_train.py --local")
        return None, None

    logger.info("Loading models from local files...")
    fare_model = xgb.XGBRegressor(n_jobs=-1)
    fare_model.load_model(fare_model_path)

    eta_model = xgb.XGBRegressor(n_jobs=-1)
    eta_model.load_model(eta_model_path)

    return fare_model, eta_model


def load_models_from_mlflow(mlflow_uri: str):
    """MLflow Registry에서 모델 로드 (Staging -> Latest 순으로 정밀 추적)"""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    def _load_single_model(model_name: str):
        logger.info(f"Attempting to load production model for [{model_name}]...")
        return mlflow.xgboost.load_model(f"models:/{model_name}@production")
  
    # 실무형 Fallback 로직 적용하여 모델 로드 (experiment_name 인자 제거)
    fare_model = _load_single_model("nyc-taxi-fare")
    eta_model  = _load_single_model("nyc-taxi-eta")
    
    return fare_model, eta_model


# ──────────────────────────────────────────────
# [2단계] 공통 예측 함수 (예측은 오직 한군데서!)
# ──────────────────────────────────────────────

def run_prediction(fare_model, eta_model, input_df: pd.DataFrame) -> dict:
    """로드된 모델 타입(Booster vs Regressor)을 자동 판별하여 한곳에서 예측 처리"""
    # 안전한 연산을 위해 모든 수치형 데이터 float로 통일
    input_df_processed = input_df.astype(float)

    # 1. Fare 요금 예측
    if isinstance(fare_model, xgb.Booster):
        dtrain = xgb.DMatrix(input_df_processed, nthread=-1)
        fare = fare_model.predict(dtrain)[0]
    else:
        fare = fare_model.predict(input_df_processed)[0]

    # 2. ETA 시간 예측
    if isinstance(eta_model, xgb.Booster):
        dtrain = xgb.DMatrix(input_df_processed, nthread=-1)
        eta = eta_model.predict(dtrain)[0]
    else:
        eta = eta_model.predict(input_df_processed)[0]

    return {"fare_usd": round(float(fare), 2), "eta_min": round(float(eta), 1)}


# ──────────────────────────────────────────────
# 메인 제어 흐름
# ──────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── [Step 1] 프로그램 시작 시 모델 로드는 단 '한 번'만 수행
    if args.mlflow_uri:
        fare_model, eta_model = load_models_from_mlflow(args.mlflow_uri)
    else:
        fare_model, eta_model = load_models_from_file(args.fare_model, args.eta_model)

    if fare_model is None or eta_model is None:
        logger.error("Model loading failed. Exiting pipeline.")
        return

    # ── 예측 입력 시나리오 데이터 정의
    scenarios = [
        {
            "name": "평일 오전 출근 (미드타운 → JFK)",
            "data": {
                "pickup_hour": 8, "pickup_dayofweek": 1, "pickup_month": 6,
                "trip_distance": 15.2, "passenger_count": 1,
                "PULocationID": 161, "DOLocationID": 132,
                "is_weekend": 0, "is_rush_hour": 1,
            },
        },
        {
            "name": "금요일 저녁 퇴근 (맨하탄 → 브루클린)",
            "data": {
                "pickup_hour": 18, "pickup_dayofweek": 4, "pickup_month": 6,
                "trip_distance": 4.8, "passenger_count": 2,
                "PULocationID": 237, "DOLocationID": 33,
                "is_weekend": 0, "is_rush_hour": 1,
            },
        },
        {
            "name": "주말 낮 관광 (센트럴파크 → 타임스퀘어)",
            "data": {
                "pickup_hour": 13, "pickup_dayofweek": 6, "pickup_month": 8,
                "trip_distance": 1.2, "passenger_count": 3,
                "PULocationID": 43, "DOLocationID": 164,
                "is_weekend": 1, "is_rush_hour": 0,
            },
        },
        {
            "name": "새벽 공항 픽업 (JFK → 맨하탄)",
            "data": {
                "pickup_hour": 2, "pickup_dayofweek": 0, "pickup_month": 3,
                "trip_distance": 18.5, "passenger_count": 1,
                "PULocationID": 132, "DOLocationID": 161,
                "is_weekend": 0, "is_rush_hour": 0,
            },
        },
    ]

    print("\n" + "=" * 65)
    print("  NYC Taxi – 요금(Fare) & ETA 예측 결과")
    print("=" * 65)

    # ── [Step 2] 반복 루프 안에서는 네트워크 요청 없이 오직 예측 연산만 빠르게 수행
    for scenario in scenarios:
        df = pd.DataFrame([scenario["data"]])
        
        # 공통 예측 함수 호출
        result = run_prediction(fare_model, eta_model, df)

        print(f"\n  📍 {scenario['name']}")
        print(f"     거리    : {scenario['data']['trip_distance']} miles")
        print(f"     예측 요금: ${result['fare_usd']:.2f}")
        print(f"     예측 ETA : {result['eta_min']:.1f}분")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi – Fare & ETA 예측 예시")
    parser.add_argument("--fare-model",  default="models/model_fare.json", help="요금 모델 파일 경로")
    parser.add_argument("--eta-model",   default="models/model_eta.json",  help="ETA 모델 파일 경로")
    parser.add_argument("--mlflow-uri",  default="http://mlflow.cnapcloud.com", help="MLflow URI (지정 시 Registry에서 로드)")
    args = parser.parse_args()
    main(args)