import pandas as pd
import re
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

# Single input + single output
INPUT_FLAT_PATH = ARTIFACTS_DIR / "canonical_flat.csv"
OUTPUT_WIDE_VALIDATED = ARTIFACTS_DIR / "canonical_wide_validated.csv"

# Validation logs
OUTPUT_ACTIONS_PATH = ARTIFACTS_DIR / "validation_actions.csv"
OUTPUT_FLAGS_PATH = ARTIFACTS_DIR / "validation_flags.csv"


# ============================================================
# Profiling + flagging
# ============================================================

def profile_column(series, sample_n=20):
    s = series.dropna().astype(str)

    return {
        "total": len(s),
        "unique": s.nunique(),
        "numeric_pct": s.str.fullmatch(r"\d+(\.\d+)?").mean() if len(s) else 0.0,
        "quantity_unit_pct": s.str.fullmatch(r"\d+(\.\d+)?\s*[a-zA-Z]+").mean() if len(s) else 0.0,
        "sample_values": s.unique()[:sample_n].tolist(),
    }


def flag_column(column, profile, path: Path):
    record = {
        "column": column,
        "issue": "mixed_patterns",
        "total_rows": profile["total"],
        "unique_values": profile["unique"],
        "numeric_pct": profile["numeric_pct"],
        "quantity_unit_pct": profile["quantity_unit_pct"],
        "sample_values": "; ".join(profile["sample_values"]),
        "flagged_at": datetime.utcnow().isoformat(),
    }

    pd.DataFrame([record]).to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )


# ============================================================
# Auto-normalization: quantity/unit split (high confidence only)
# ============================================================

def normalize_quantity_unit_auto(df, confidence=0.98):
    col = "product.quantity_produced"
    if col not in df.columns:
        return df, None

    profile = profile_column(df[col])

    if profile["quantity_unit_pct"] >= confidence:
        qty, unit = [], []

        for v in df[col]:
            if pd.isna(v):
                qty.append(pd.NA)
                unit.append(pd.NA)
            else:
                m = re.match(r"^\s*(\d+(\.\d+)?)\s*([a-zA-Z]+)\s*$", str(v))
                if m:
                    qty.append(m.group(1))
                    unit.append(m.group(3))
                else:
                    qty.append(pd.NA)
                    unit.append(pd.NA)

        df = df.copy()
        df[col] = qty

        if "product.quantity_unit" not in df.columns:
            df["product.quantity_unit"] = unit

        return df, {
            "column": col,
            "action": "auto-normalized quantity/unit split",
            "confidence": float(profile["quantity_unit_pct"]),
            "sample_values": "; ".join(profile["sample_values"]),
        }

    return df, None


# ============================================================
# Metrics + prune (incorporated from calc_metrics_and_prune.py)
# ============================================================

def _to_num(s: pd.Series) -> pd.Series:
    """Coerce strings like '60', '60 min', '' to floats; blanks -> NaN."""
    if s is None:
        return pd.Series(dtype="float64")

    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")

    def parse_one(v):
        if pd.isna(v):
            return np.nan
        txt = str(v).strip()
        if txt == "" or txt.lower() in {"nan", "none", "null"}:
            return np.nan
        m = re.search(r"[-+]?\d*\.?\d+", txt)
        return float(m.group(0)) if m else np.nan

    return s.map(parse_one).astype("float64")


def normalize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure date/start/end columns are datetime64[ns] if present.
    We interpret start/end as datetimes (date component may be defaulted) for type safety.
    """
    out = df.copy()

    if "production.production_date" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["production.production_date"]):
        out["production.production_date"] = pd.to_datetime(
            out["production.production_date"].astype(str).str.strip(),
            errors="coerce",
        )

    if "timing.start_time" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["timing.start_time"]):
        out["timing.start_time"] = pd.to_datetime(
            out["timing.start_time"].astype(str).str.strip(),
            errors="coerce",
        )

    if "timing.end_time" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["timing.end_time"]):
        out["timing.end_time"] = pd.to_datetime(
            out["timing.end_time"].astype(str).str.strip(),
            errors="coerce",
        )

    return out


def compute_metrics(df: pd.DataFrame):
    """
    Computes/ensures these columns exist if inputs are available:
      - run_minutes
      - total_downtime_minutes
      - operator_error_minutes (defaults to 0 unless already present)
      - overrun_minutes
      - efficiency_pct
    Returns (df, action_dict_or_None)
    """
    out = normalize_datetime_columns(df)
    did_any = False
    notes = []

    # ---------- run_minutes ----------
    if "run_minutes" not in out.columns:
        if {"timing.start_time", "timing.end_time"}.issubset(out.columns):
            start = out["timing.start_time"]
            end = out["timing.end_time"]

            # midnight rollover
            end2 = end + pd.to_timedelta((end < start).astype("int64"), unit="D")

            out["run_minutes"] = ((end2 - start).dt.total_seconds() / 60.0).clip(lower=0)
            did_any = True
            notes.append("run_minutes")
        else:
            out["run_minutes"] = 0.0
            did_any = True
            notes.append("run_minutes_defaulted")

    # ---------- total_downtime_minutes ----------
    if "total_downtime_minutes" not in out.columns:
        downtime_cols = [c for c in out.columns if c.startswith("downtime.factor_")]
        if downtime_cols:
            out["total_downtime_minutes"] = out[downtime_cols].apply(_to_num).sum(axis=1, skipna=True)
            did_any = True
            notes.append("total_downtime_minutes")
        else:
            out["total_downtime_minutes"] = 0.0
            did_any = True
            notes.append("total_downtime_minutes_defaulted")

    # ---------- operator_error_minutes ----------
    # Can't infer from flat alone unless already present, so default.
    if "operator_error_minutes" not in out.columns:
        out["operator_error_minutes"] = 0.0
        did_any = True
        notes.append("operator_error_minutes_defaulted")

    # ---------- overrun_minutes ----------
    if "overrun_minutes" not in out.columns:
        if "timing.min_run_minutes" in out.columns:
            min_run = _to_num(out["timing.min_run_minutes"]).fillna(0)
            out["overrun_minutes"] = (_to_num(out["run_minutes"]).fillna(0) - min_run).clip(lower=0)
            did_any = True
            notes.append("overrun_minutes")
        else:
            out["overrun_minutes"] = 0.0
            did_any = True
            notes.append("overrun_minutes_defaulted")

    # ---------- efficiency_pct ----------
    if "efficiency_pct" not in out.columns:
        run_m = _to_num(out["run_minutes"]).fillna(0)
        dt_m = _to_num(out["total_downtime_minutes"]).fillna(0)
        denom = run_m.replace({0: np.nan})
        out["efficiency_pct"] = (((run_m - dt_m).clip(lower=0)) / denom * 100.0).fillna(0)
        did_any = True
        notes.append("efficiency_pct")

    action = None
    if did_any:
        action = {
            "column": "metrics.*",
            "action": "computed derived metrics",
            "confidence": 1.0,
            "sample_values": "; ".join(notes),
        }

    return out, action


def drop_empty_columns(df: pd.DataFrame):
    """
    Drops columns that are entirely empty:
      - all values NaN, OR
      - for object cols: all values blank/whitespace/'nan'/'null'/'none'
    Does NOT drop all-zero columns.
    Returns (df, action_dict_or_None)
    """
    out = df.copy()
    to_drop = []

    for c in out.columns:
        s = out[c]

        if s.isna().all():
            to_drop.append(c)
            continue

        if s.dtype == "object":
            cleaned = s.astype(str).str.strip().str.lower()
            cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
            if cleaned.isna().all():
                to_drop.append(c)

    if not to_drop:
        return out, None

    out = out.drop(columns=to_drop)
    return out, {
        "column": "columns.*",
        "action": f"pruned empty columns ({len(to_drop)})",
        "confidence": 1.0,
        "sample_values": "; ".join(to_drop[:25]) + ("; ..." if len(to_drop) > 25 else ""),
    }


# ============================================================
# Validate canonical_flat -> canonical_wide_validated
# ============================================================

def validate_canonical_flat(df: pd.DataFrame):
    actions = []

    # Quantity normalization
    df, action = normalize_quantity_unit_auto(df)
    if action:
        actions.append(action)
    elif "product.quantity_produced" in df.columns:
        profile = profile_column(df["product.quantity_produced"])
        flag_column("product.quantity_produced", profile, OUTPUT_FLAGS_PATH)

    # Metrics
    df, m_action = compute_metrics(df)
    if m_action:
        actions.append(m_action)

    # Prune
    df, p_action = drop_empty_columns(df)
    if p_action:
        actions.append(p_action)

    return df, actions


# ============================================================
# MAIN
# ============================================================

def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FLAT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FLAT_PATH}")

    print(f"Loading: {INPUT_FLAT_PATH.name}")
    df = pd.read_csv(INPUT_FLAT_PATH, dtype=str)

    df, actions = validate_canonical_flat(df)

    df.to_csv(OUTPUT_WIDE_VALIDATED, index=False)
    print(f"Wrote: {OUTPUT_WIDE_VALIDATED.name}")

    if actions:
        pd.DataFrame(actions).to_csv(OUTPUT_ACTIONS_PATH, index=False)
        print(f"Wrote: {OUTPUT_ACTIONS_PATH.name}")
    else:
        print("No auto-normalizations applied")

    print("Validation complete")


if __name__ == "__main__":
    main()
