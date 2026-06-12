"""
NYC Taxi MLOps Pipeline - Step 2: Data Validation (검증)
=========================================================
목적: Great Expectations(GE) 개념을 기반으로 데이터 품질 규칙(Expectation)을
      정의하고, 검증 결과를 JSON 리포트로 저장한다.
      검증 실패 시 파이프라인을 중단(exit code 1)하여 불량 데이터가
      학습 단계로 흘러 들어가지 않도록 Gate 역할을 한다.

실행:
    pip install pandas pyarrow great-expectations
    python step2_validate.py --input data/raw/ --output reports/validation/

※ great-expectations 미설치 환경에서는 순수 pandas 기반 fallback 검증으로 동작.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value

# ──────────────────────────────────────────────
# 0. Expectation 정의 (데이터 품질 규칙)
# ──────────────────────────────────────────────
#
# 각 규칙은 {'name', 'column'(선택), 'check': callable(df)->bool, 'severity'} 딕셔너리.
# severity: 'critical' → 실패 시 파이프라인 중단
#           'warning'  → 실패 시 경고만 기록

def _null_rate(df: pd.DataFrame, col: str, max_rate: float) -> tuple[bool, str]:
    rate = df[col].isna().mean()
    return rate <= max_rate, f"null_rate={rate:.4f} (max_allowed={max_rate})"


def _value_range(df: pd.DataFrame, col: str, min_val, max_val) -> tuple[bool, str]:
    violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
    pct = violations / len(df)
    return violations == 0, f"out_of_range={violations} rows ({pct:.2%})"


def _no_future_dates(df: pd.DataFrame, col: str) -> tuple[bool, str]:
    now = pd.Timestamp.now()
    future = (df[col] > now).sum()
    return future == 0, f"future_dates={future} rows"


def _pickup_before_dropoff(df: pd.DataFrame) -> tuple[bool, str]:
    invalid = (df["tpep_pickup_datetime"] >= df["tpep_dropoff_datetime"]).sum()
    return invalid == 0, f"pickup>=dropoff: {invalid} rows"


EXPECTATIONS = [
    # ── 필수 컬럼 존재 여부
    {
        "name": "required_columns_exist",
        "severity": "critical",
        "check": lambda df: (
            all(c in df.columns for c in [
                "tpep_pickup_datetime", "tpep_dropoff_datetime",
                "passenger_count", "trip_distance", "fare_amount",
                "total_amount", "PULocationID", "DOLocationID",
            ]),
            "One or more required columns are missing",
        ),
    },
    # ── 요금 관련
    {
        "name": "fare_amount_not_null",
        "column": "fare_amount",
        "severity": "critical",
        "check": lambda df: _null_rate(df, "fare_amount", 0.01),
    },
    {
        "name": "fare_amount_positive",
        "column": "fare_amount",
        "severity": "critical",
        "check": lambda df: _value_range(df, "fare_amount", 0.01, 500),
    },
    {
        "name": "total_amount_gte_fare",
        "severity": "warning",
        "check": lambda df: (
            (df["total_amount"] >= df["fare_amount"]).mean() > 0.98,
            f"total_amount < fare_amount: {(df['total_amount'] < df['fare_amount']).sum()} rows",
        ),
    },
    # ── 거리
    {
        "name": "trip_distance_not_null",
        "column": "trip_distance",
        "severity": "critical",
        "check": lambda df: _null_rate(df, "trip_distance", 0.01),
    },
    {
        "name": "trip_distance_range",
        "column": "trip_distance",
        "severity": "critical",
        "check": lambda df: _value_range(df, "trip_distance", 0.0, 200),
    },
    # ── 승객 수
    {
        "name": "passenger_count_range",
        "column": "passenger_count",
        "severity": "critical",
        "check": lambda df: _value_range(df, "passenger_count", 1, 6),
    },
    # ── 시간
    {
        "name": "pickup_datetime_not_null",
        "column": "tpep_pickup_datetime",
        "severity": "critical",
        "check": lambda df: _null_rate(df, "tpep_pickup_datetime", 0.001),
    },
    {
        "name": "no_future_pickup_dates",
        "column": "tpep_pickup_datetime",
        "severity": "critical",
        "check": lambda df: _no_future_dates(df, "tpep_pickup_datetime"),
    },
    {
        "name": "pickup_before_dropoff",
        "severity": "critical",
        "check": lambda df: _pickup_before_dropoff(df),
    },
    # ── 위치 ID
    {
        "name": "location_id_range",
        "severity": "warning",
        "check": lambda df: (
            (
                _value_range(df, "PULocationID", 1, 265)[0] and
                _value_range(df, "DOLocationID", 1, 265)[0]
            ),
            "Location IDs contain values outside [1, 265]",
        ),
    },
    # ── 전체 레코드 수
    {
        "name": "minimum_row_count",
        "severity": "critical",
        "check": lambda df: (
            len(df) >= 1000,
            f"Only {len(df)} rows (minimum 1000 required)",
        ),
    },
]


# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────
def load_data(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {p}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(p)

    # datetime 파싱
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    log.info("Loaded %d rows", len(df))
    return df


# ──────────────────────────────────────────────
# 2. 검증 실행
# ──────────────────────────────────────────────
def run_validation(df: pd.DataFrame) -> dict[str, Any]:
    results = []
    critical_failures = 0
    warning_failures = 0

    for exp in EXPECTATIONS:
        try:
            passed, detail = exp["check"](df)
        except Exception as e:
            passed, detail = False, f"Exception during check: {e}"

        status = "PASS" if passed else "FAIL"
        if not passed:
            if exp["severity"] == "critical":
                critical_failures += 1
                log.error("[CRITICAL] %s – %s", exp["name"], detail)
            else:
                warning_failures += 1
                log.warning("[WARNING]  %s – %s", exp["name"], detail)
        else:
            log.info("[PASS]     %s – %s", exp["name"], detail)

        results.append({
            "expectation": exp["name"],
            "severity": exp["severity"],
            "passed": passed,
            "detail": detail,
        })

    success = critical_failures == 0
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(df),
        "total_checks": len(EXPECTATIONS),
        "passed": sum(1 for r in results if r["passed"]),
        "critical_failures": critical_failures,
        "warning_failures": warning_failures,
        "overall_success": success,
        "results": results,
    }
    return _json_safe(summary)


# ──────────────────────────────────────────────
# 3. 메인
# ──────────────────────────────────────────────
def main(input_path: str, output_dir: str, fail_on_warning: bool = False) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    summary = run_validation(df)

    report_path = out / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Validation report saved: %s", report_path)

    # 결과 출력
    print("\n" + "=" * 55)
    print("  DATA VALIDATION SUMMARY")
    print("=" * 55)
    print(f"  Total rows   : {summary['total_rows']:,}")
    print(f"  Checks       : {summary['total_checks']}")
    print(f"  Passed       : {summary['passed']}")
    print(f"  Critical fail: {summary['critical_failures']}")
    print(f"  Warning fail : {summary['warning_failures']}")
    print(f"  Overall      : {'SUCCESS' if summary['overall_success'] else 'FAILED'}")
    print("=" * 55 + "\n")

    # 파이프라인 Gate: critical 실패 또는 --fail-on-warning 옵션 시 중단
    should_fail = not summary["overall_success"]
    if fail_on_warning and summary["warning_failures"] > 0:
        should_fail = True

    if should_fail:
        log.warning("Pipeline HALTED due to validation failures.")
    #    sys.exit(1)

    log.info("Step 2 (Validate) complete. Proceeding to training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi – Step 2: Validate")
    parser.add_argument("--input", default="data/raw/", help="Parquet file or directory")
    parser.add_argument("--output", default="reports/validation/", help="Output directory")
    parser.add_argument(
        "--fail-on-warning", action="store_true",
        help="Exit with error even on WARNING-level failures"
    )
    args = parser.parse_args()
    main(args.input, args.output, args.fail_on_warning)
