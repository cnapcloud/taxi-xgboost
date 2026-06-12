"""
NYC Taxi MLOps Pipeline - Orchestrator
=======================================
전체 5단계를 순차 실행한다.

실행:
    python pipeline.py --input data/raw/ --mode local
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
REPORTS    = BASE_DIR / "reports"
MODELS     = BASE_DIR / "models"
CHAMPION   = BASE_DIR / "models" / "champion"


def run_step(name: str, cmd: list[str], allowed_exit_codes: list[int] = None) -> int:
    allowed = allowed_exit_codes or [0]
    log.info("▶ %s", name)
    result = subprocess.run(cmd)
    if result.returncode not in allowed:
        log.error("✗ %s FAILED (exit=%d)", name, result.returncode)
        sys.exit(result.returncode)
    log.info("✓ %s done (exit=%d)", name, result.returncode)
    return result.returncode


def main(args: argparse.Namespace) -> None:
    start = datetime.now()
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # Step 1: 분석
    run_step("Step 1: Analyze", [
        py, str(BASE_DIR / "step1_analyze.py"),
        "--input", args.input, "--output", str(REPORTS / "analysis"),
    ])

    # Step 2: 검증
    run_step("Step 2: Validate", [
        py, str(BASE_DIR / "step2_validate.py"),
        "--input", args.input, "--output", str(REPORTS / "validation"),
    ])

    # Step 3: 학습 (Fare + ETA)
    train_cmd = [
        py, str(BASE_DIR / "step3_train.py"),
        "--input", args.input,
        "--mlflow-uri", args.mlflow_uri,
    ]
    if args.mode == "local":
        train_cmd.append("--local")
    else:
        train_cmd += ["--ray-address", args.ray_address, "--num-workers", str(args.num_workers)]
    run_step("Step 3: Train (Fare + ETA)", train_cmd)

    # Step 4: 평가 (Fare + ETA 각각 챔피언 비교)
    eval_cmd = [
        py, str(BASE_DIR / "step4_evaluate.py"),
        "--test-data", args.input,
        "--threshold", str(args.threshold),
    ]

    code = run_step("Step 4: Evaluate", eval_cmd, allowed_exit_codes=[0, 2])
    if code == 2:
        log.warning("Challenger(s) did not improve. Pipeline stopped. Current models kept.")
        sys.exit(0)

    # Step 5: 등록 (Fare + ETA 각각 Registry 등록)
    run_step("Step 5: Register", [
        py, str(BASE_DIR / "step5_register.py"),
        "--mlflow-uri", args.mlflow_uri,
        "--auto-promote",
    ])

    elapsed = (datetime.now() - start).total_seconds()
    log.info("✅ Pipeline COMPLETE in %.1fs", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default="data/raw/")
    parser.add_argument("--mode",         choices=["local", "distributed"], default="local")
    parser.add_argument("--ray-address",  default=None, help="Ray cluster address (e.g., ray://localhost:10001)")
    parser.add_argument("--num-workers",  type=int,   default=1)
    parser.add_argument("--n-estimators", type=int,   default=300)
    parser.add_argument("--mlflow-uri",   default="http://mlflow.cnapcloud.com")
    parser.add_argument("--threshold",    type=float, default=0.0)
    args = parser.parse_args()
    main(args)
