import pandas as pd
import pytest
from pathlib import Path


# ============================================================
# PATH SETUP (WSL-SAFE, PYTEST-SAFE)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]  # Capstone--Operator-Selection-Pipeline

ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

CANONICAL_FLAT_PATH = ARTIFACTS_DIR / "canonical_flat.csv"


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def canonical_flat():
    if not CANONICAL_FLAT_PATH.exists():
        raise FileNotFoundError(
            f"Required test input not found: {CANONICAL_FLAT_PATH}"
        )
    return pd.read_csv(CANONICAL_FLAT_PATH)


# ============================================================
# Tests
# ============================================================

def test_unique_run_id(canonical_flat):
    assert canonical_flat["run_id_canonical"].is_unique, (
        "run_id_canonical must be unique (1 row per run)"
    )


def test_no_missing_core_fields(canonical_flat):
    required = [
        "run_id_canonical",
        "production.production_date",
        "timing.start_time",
        "timing.end_time",
        "run_minutes",
    ]
    missing = canonical_flat[required].isna().any()
    assert not missing.any(), (
        f"Missing values in core fields: "
        f"{missing[missing].index.tolist()}"
    )


def test_run_minutes_positive(canonical_flat):
    assert (canonical_flat["run_minutes"] > 0).all(), (
        "run_minutes must be strictly positive"
    )


def test_downtime_not_exceed_run(canonical_flat):
    assert (
        canonical_flat["total_downtime_minutes"]
        <= canonical_flat["run_minutes"]
    ).all(), "Downtime exceeds run duration"


def test_operator_error_not_exceed_downtime(canonical_flat):
    assert (
        canonical_flat["operator_error_minutes"]
        <= canonical_flat["total_downtime_minutes"]
    ).all(), "Operator error minutes exceed total downtime"


def test_downtime_factors_sum(canonical_flat):
    factor_cols = [
        c for c in canonical_flat.columns
        if c.startswith("downtime.factor_")
    ]
    summed = canonical_flat[factor_cols].fillna(0).sum(axis=1)
    assert (
        summed == canonical_flat["total_downtime_minutes"]
    ).all(), (
        "Sum of downtime factors does not equal total_downtime_minutes"
    )


def test_min_run_constraint(canonical_flat):
    assert (
        canonical_flat["run_minutes"]
        >= canonical_flat["timing.min_run_minutes"]
    ).all(), (
        "Run duration is below product minimum run time"
    )
