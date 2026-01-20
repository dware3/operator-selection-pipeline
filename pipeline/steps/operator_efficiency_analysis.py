# ============================================================
# operator_efficiency_analysis.py
# Single-input version: canonical_wide_validated.csv only
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

# ============================================================
# PATH SETUP (WSL-SAFE, NO CWD ASSUMPTIONS)
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

INPUT_VALIDATED_PATH = ARTIFACTS_DIR / "canonical_wide_validated.csv"


# ============================================================
# Load + prepare (wide -> long operator)
# ============================================================

def load_and_prepare(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    # Identify operator columns in wide format
    operator_cols = [c for c in df.columns if c.endswith("_operator")]
    if not operator_cols:
        raise ValueError(
            "No operator columns found in canonical_wide_validated.csv. "
            "Expected columns like 'stations.station_1_operator'."
        )

    required_base = {"run_minutes"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure operator_error_minutes exists (0 if absent)
    if "operator_error_minutes" not in df.columns:
        df["operator_error_minutes"] = 0

    # Melt wide operators into a single operator column
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

    # Clean types
    df["run_minutes"] = pd.to_numeric(df["run_minutes"], errors="coerce")
    df["operator_error_minutes"] = pd.to_numeric(df["operator_error_minutes"], errors="coerce").fillna(0)

    df = df[df["run_minutes"] > 0].copy()

    # Efficiency definition (your existing intent)
    df["efficiency"] = (df["run_minutes"] - df["operator_error_minutes"]).clip(lower=0) / df["run_minutes"]

    return df


# ------------------------------------------------------------
# Method 1: Linear regression (operator as categorical)
# ------------------------------------------------------------

def operator_regression(df: pd.DataFrame):
    model = smf.ols("efficiency ~ C(operator)", data=df).fit()
    print("\n=== Linear Regression: Operator → Efficiency ===\n")
    print(model.summary())


# ------------------------------------------------------------
# Method 2: One-way ANOVA
# ------------------------------------------------------------

def operator_anova(df: pd.DataFrame):
    groups = [g["efficiency"].values for _, g in df.groupby("operator")]
    if len(groups) < 2:
        print("\n=== One-way ANOVA ===")
        print("Not enough operator groups for ANOVA.")
        return

    f_stat, p_value = stats.f_oneway(*groups)

    print("\n=== One-way ANOVA ===")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value:     {p_value:.4f}")

    if p_value < 0.05:
        print("→ Operator differences are unlikely due to chance")
    else:
        print("→ No statistically clear operator effect")


# ------------------------------------------------------------
# Method 3: Permutation test
# ------------------------------------------------------------

def permutation_test(df: pd.DataFrame, n_iter: int = 10_000):
    real_var = df.groupby("operator")["efficiency"].mean().var()

    permuted_vars = []
    for _ in range(n_iter):
        shuffled = df.copy()
        shuffled["operator"] = np.random.permutation(shuffled["operator"].values)
        permuted_vars.append(shuffled.groupby("operator")["efficiency"].mean().var())

    permuted_vars = np.array(permuted_vars)
    p_value = np.mean(permuted_vars >= real_var)

    print("\n=== Permutation Test ===")
    print(f"Observed operator variance: {real_var:.6f}")
    print(f"P-value:                   {p_value:.4f}")

    if p_value < 0.05:
        print("→ Operator–efficiency link is very unlikely to be random")
    else:
        print("→ Observed differences are consistent with noise")


# ============================================================
# Main
# ============================================================

def main():
    print(f"Loading validated canonical data: {INPUT_VALIDATED_PATH}")
    df = load_and_prepare(INPUT_VALIDATED_PATH)

    operator_regression(df)
    operator_anova(df)
    permutation_test(df)


if __name__ == "__main__":
    main()
