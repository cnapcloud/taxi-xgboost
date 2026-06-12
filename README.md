# NYC Taxi MLOps Pipeline

NYC Yellow Taxi 데이터셋 기반 End-to-End MLOps 파이프라인 예제입니다.
택시 요금 예측(Regression)을 통해 각 단계의 존재 이유를 명확히 설명합니다.

## 파이프라인 구조

```
[Step 1: 분석]  →  [Step 2: 검증]  →  [Step 3: 학습]  →  [Step 4: 평가]  →  [Step 5: 등록]
 EDA 리포트       데이터 품질 Gate    KubeRay 분산 XGB   Champion 비교      MLflow Registry
 시각화 생성       실패 시 중단        MLflow 실험 추적    실패 시 중단        Production 전환
```

## 각 파일 설명

| 파일 | 단계 | 주요 라이브러리 |
|------|------|----------------|
| `step1_analyze.py` | 분석 (Analyze) | pandas, matplotlib, seaborn |
| `step2_validate.py` | 검증 (Validate) | pandas (Great Expectations 개념) |
| `step3_train.py` | 학습 (Train) | xgboost, ray[train], mlflow |
| `step4_evaluate.py` | 평가 (Evaluate) | xgboost, scikit-learn |
| `step5_register.py` | 등록 (Register) | mlflow, xgboost |
| `pipeline.py` | 오케스트레이터 | subprocess |

## 실행 방법

### 1. 데이터 준비

```bash
mkdir -p data/raw
# NYC Taxi 공식 데이터 다운로드 (2024년 1월, ~500MB)
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet \
     -O data/raw/yellow_tripdata_2024-01.parquet
```

### 2. 패키지 설치

```bash
python3 -m venv .venv
pip install -r requirements.txt
```

### 3. 전체 파이프라인 실행 (로컬)

```bash
python pipeline.py --input data/raw/ --mode local --mlflow-uri http://localhost:5000
```

### 4. 단계별 실행

```bash
# Step 1: 분석
python step1_analyze.py --input data/raw/ --output reports/

# Step 2: 검증 (실패 시 exit code 1 → CI/CD 파이프라인 중단)
python step2_validate.py --input data/raw/ --output reports/validation/

# Step 3: 학습 (로컬)
python step3_train.py --input data/raw/ --output models/ --local

# Step 3: 학습 (KubeRay 분산)
python step3_train.py \
  --input data/raw/ \
  --mlflow-uri http://mlflow.cnapcloud.com \
  --local

# Step 4: 평가
python step4_evaluate.py \
  --mlflow-uri http://mlflow.cnapcloud.com \
  --output reports/evaluation/ \
  --threshold 0.0

# Step 5: 등록
python step5_register.py \
  --mlflow-uri http://mlflow.cnapcloud.com \
  --auto-promote
```

## 파이프라인 Gate 동작

| 단계 | 실패 조건 | Exit Code | 동작 |
|------|-----------|-----------|------|
| Step 2 | Critical 검증 실패 | 1 | 파이프라인 중단 |
| Step 4 | 챌린저 RMSE 개선 미달 | 2 | 파이프라인 중단 (기존 모델 유지) |
| Step 5 | Smoke test 실패 | 1 | Production 롤백 필요 |
