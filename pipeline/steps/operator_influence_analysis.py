# ============================================================
# operator_influence_analysis.py
# Single-input version: canonical_wide_validated.csv only
# ============================================================

import pandas as pd
from scipy import stats
from pathlib import Path

# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

INPUT_WIDE = ARTIFACTS_DIR / "canonical_wide_validated.csv"

OUTPUT_FACTOR_TESTS = ARTIFACTS_DIR / "operator_factor_tests.csv"
OUTPUT_OPERATOR_PRODUCT = ARTIFACTS_DIR / "operator_product_downtime.csv"
OUTPUT_VARIANCE = ARTIFACTS_DIR / "operator_downtime_variability.csv"

DOWNTIME_PREFIX = "downtime.factor_"


# ============================================================
# Load & prepare (wide -> long operator)
# ============================================================

def load_and_prepare_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Loading {path.name}")
    df = pd.read_csv(path)

    factor_cols = [c for c in df.columns if c.startswith(DOWNTIME_PREFIX)]
    if factor_cols:
        df[factor_cols] = df[factor_cols].fillna(0)

    operator_cols = [c for c in df.columns if c.endswith("_operator")]
    if not operator_cols:
        raise ValueError(
            f"{path.name} has no operator columns (expected '*_operator'). "
            "If you intended stations-long, that file is no longer supported here."
        )

    # Melt operators
    id_vars = [c for c in df.columns if c not in operator_cols]
    df = (
        df.melt(
            id_vars=id_vars,
            value_vars=operator_cols,
            var_name="station",
            value_name="operator",
        )
        .dropna(subset=["operator"])
        .copy()
    )

    required = {"operator", "run_minutes", "total_downtime_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")

    df["run_minutes"] = pd.to_numeric(df["run_minutes"], errors="coerce")
    df["total_downtime_minutes"] = pd.to_numeric(df["total_downtime_minutes"], errors="coerce").fillna(0)

    df = df[df["run_minutes"] > 0].copy()
    df["downtime_rate"] = df["total_downtime_minutes"] / df["run_minutes"]

    return df


# ============================================================
# Analysis
# ============================================================

def operator_factor_tests(df: pd.DataFrame) -> pd.DataFrame:
    factors = [c for c in df.columns if c.startswith(DOWNTIME_PREFIX)]
    rows = []

    for f in factors:
        groups = [g[f].values for _, g in df.groupby("operator") if g[f].sum() > 0]
        if len(groups) < 2:
            continue
        f_stat, p = stats.f_oneway(*groups)
        rows.append((f, f_stat, p))

    if not rows:
        return pd.DataFrame(columns=["downtime_factor", "f_stat", "p_value"])

    return (
        pd.DataFrame(rows, columns=["downtime_factor", "f_stat", "p_value"])
        .sort_values("p_value")
    )


def operator_product_interaction(df: pd.DataFrame) -> pd.DataFrame:
    if "product.product_code" not in df.columns:
        raise ValueError("Missing product.product_code column required for operator-product interaction.")
    return (
        df.groupby(["operator", "product.product_code"])["total_downtime_minutes"]
        .mean()
        .unstack()
    )


def operator_variance_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("operator")["downtime_rate"]
        .agg(mean="mean", std="std", runs="count")
        .sort_values("std")
    )


# ============================================================
# Main
# ============================================================

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_wide(INPUT_WIDE)

    operator_factor_tests(df).to_csv(OUTPUT_FACTOR_TESTS, index=False)
    operator_product_interaction(df).to_csv(OUTPUT_OPERATOR_PRODUCT)
    operator_variance_analysis(df).to_csv(OUTPUT_VARIANCE)

    print("Operator influence analysis complete")


if __name__ == "__main__":
    main()
