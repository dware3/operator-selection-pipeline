# operator_assignment_interactive.py
#
# Interactive Operator Assignment Wizard
# - Single input: pipeline/artifacts/canonical_wide_validated.csv
# - You choose the "assignment field" (e.g., product.product_code)
# - Builds operator assignments based on lowest downtime-rate variability
# - Outputs: pipeline/artifacts/<assignment_field>_operator_assignments.csv

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

INPUT_PATH = ARTIFACTS_DIR / "canonical_wide_validated.csv"


# ============================================================
# CONFIG
# ============================================================

MIN_RUNS_HIGH = 8
MIN_RUNS_MED = 5


# ============================================================
# UTILS
# ============================================================

def safe_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s.strip("_") or "assignment"


def pick_from_list(prompt: str, options: list[str]) -> str:
    if not options:
        raise ValueError("No options available to select.")
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    while True:
        choice = input("> ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(options):
            return options[int(choice)]
        print("Enter a valid number from the list.")


def yes_no(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    v = input(f"{prompt} [{d}]: ").strip().lower()
    if not v:
        return default
    return v in {"y", "yes", "1", "true"}


# ============================================================
# LOAD + PREPARE
# ============================================================

def load_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)

    required = {"run_id_canonical", "run_minutes", "total_downtime_minutes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    operator_cols = [c for c in df.columns if c.startswith("stations.") and c.endswith("_operator")]
    if not operator_cols:
        raise ValueError("No operator columns found (expected stations.*_operator).")

    df["run_minutes"] = pd.to_numeric(df["run_minutes"], errors="coerce")
    df["total_downtime_minutes"] = pd.to_numeric(df["total_downtime_minutes"], errors="coerce").fillna(0)

    df = df[df["run_minutes"] > 0].copy()
    df["downtime_rate"] = df["total_downtime_minutes"] / df["run_minutes"]

    # Melt operator columns -> one operator column
    long = (
        df.melt(
            id_vars=[c for c in df.columns if c not in operator_cols],
            value_vars=operator_cols,
            var_name="station_col",
            value_name="operator",
        )
        .dropna(subset=["operator"])
        .copy()
    )

    return df, long, operator_cols


def candidate_assignment_fields(df: pd.DataFrame, operator_cols: list[str]) -> list[str]:
    # Exclude obvious non-assignment columns
    exclude_prefixes = ("downtime.",)
    exclude_exact = {
        "run_id_canonical",
        "run_minutes",
        "total_downtime_minutes",
        "operator_error_minutes",
        "overrun_minutes",
        "efficiency_pct",
        "downtime_rate",
    }
    candidates = []
    for c in df.columns:
        if c in operator_cols:
            continue
        if c in exclude_exact:
            continue
        if c.startswith(exclude_prefixes):
            continue
        # Keep columns that have at least 2 distinct non-null values (so assignment is meaningful)
        nunq = df[c].dropna().nunique()
        if nunq >= 2:
            candidates.append(c)
    # Prefer product/product_code-ish names first
    candidates.sort(key=lambda x: (0 if "product" in x else 1, x))
    return candidates


# ============================================================
# VARIABILITY + ASSIGNMENT
# ============================================================

def build_variability(long_df: pd.DataFrame, assignment_field: str) -> pd.DataFrame:
    if assignment_field not in long_df.columns:
        raise ValueError(f"Assignment field not found in data: {assignment_field}")

    # drop null assignment field (can't assign)
    d = long_df.dropna(subset=[assignment_field, "operator"]).copy()

    summary = (
        d.groupby([assignment_field, "operator"])
        .agg(
            runs=("run_id_canonical", "count"),
            std_downtime_rate=("downtime_rate", "std"),
            mean_downtime_rate=("downtime_rate", "mean"),
        )
        .reset_index()
    )

    # std is NaN for single-run groups; treat as "bad" so it won't win
    summary["std_downtime_rate"] = summary["std_downtime_rate"].fillna(np.inf)
    return summary


def greedy_one_to_one(summary: pd.DataFrame, assignment_field: str) -> pd.DataFrame:
    summary = summary.sort_values("std_downtime_rate")

    assigned_left = set()      # assignment values (e.g., product codes)
    assigned_ops = set()       # operators
    rows = []

    all_left = summary[assignment_field].unique().tolist()

    for _, r in summary.iterrows():
        left = r[assignment_field]
        op = r["operator"]

        if left in assigned_left:
            continue
        if op in assigned_ops:
            continue

        runs = int(r["runs"])
        if runs >= MIN_RUNS_HIGH:
            confidence = "high"
        elif runs >= MIN_RUNS_MED:
            confidence = "moderate"
        elif runs >= 2:
            confidence = "low"
        else:
            confidence = "very low"

        rows.append({
            assignment_field: left,
            "assigned_operator": op,
            "std_downtime_rate": None if not np.isfinite(r["std_downtime_rate"]) else round(float(r["std_downtime_rate"]), 6),
            "mean_downtime_rate": None if pd.isna(r["mean_downtime_rate"]) else round(float(r["mean_downtime_rate"]), 6),
            "runs": runs,
            "confidence": confidence,
            "explanation": "Assigned based on lowest downtime-rate variability.",
        })

        assigned_left.add(left)
        assigned_ops.add(op)

    # mark unassigned
    for left in all_left:
        if left not in assigned_left:
            rows.append({
                assignment_field: left,
                "assigned_operator": "unassigned",
                "std_downtime_rate": None,
                "mean_downtime_rate": None,
                "runs": 0,
                "confidence": "none",
                "explanation": "No available operator remaining.",
            })

    return pd.DataFrame(rows)


def top_k_per_group(summary: pd.DataFrame, assignment_field: str, k: int = 3) -> pd.DataFrame:
    # If you don’t want one-to-one, this gives “best operators per product”
    out = (
        summary.sort_values(["std_downtime_rate", "mean_downtime_rate"])
        .groupby(assignment_field, as_index=False)
        .head(k)
        .copy()
    )
    out["confidence"] = out["runs"].apply(
        lambda runs: "high" if runs >= MIN_RUNS_HIGH else
                    "moderate" if runs >= MIN_RUNS_MED else
                    "low" if runs >= 2 else
                    "very low"
    )
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading: {INPUT_PATH}")
    wide, long, operator_cols = load_wide(INPUT_PATH)

    candidates = candidate_assignment_fields(wide, operator_cols)
    if not candidates:
        raise ValueError("No suitable assignment-field candidates found in the wide file.")

    assignment_field = pick_from_list("Select the assignment dimension:", candidates)

    one_to_one = yes_no("Enforce one-to-one assignment (each operator used at most once)?", default=True)

    summary = build_variability(long, assignment_field)

    # Quick sanity stats
    print(f"\nUsing assignment field: {assignment_field}")
    print(f"Groups (unique): {summary[assignment_field].nunique()} | Operators: {summary['operator'].nunique()}")

    if one_to_one:
        result = greedy_one_to_one(summary, assignment_field)
        out_name = f"{safe_filename(assignment_field)}_operator_assignments.csv"
    else:
        k = 3
        try:
            k_in = input("How many top operators per group? [3]: ").strip()
            if k_in:
                k = max(1, int(k_in))
        except Exception:
            k = 3
        result = top_k_per_group(summary, assignment_field, k=k)
        out_name = f"{safe_filename(assignment_field)}_top_{k}_operators.csv"

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / out_name
    result.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print("\nPreview:")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
