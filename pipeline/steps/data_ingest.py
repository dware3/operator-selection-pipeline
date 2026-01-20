import re
import pandas as pd
import yaml
from pathlib import Path

# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

CONFIGS_DIR = PROJECT_ROOT / "configs"
USER_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

OUTPUT_MAPPING_PATH = ARTIFACTS_DIR / "metadata_mapping.yml"


# ============================================================
# Mapping/source_file normalization
# ============================================================

def normalize_source_file(value: str) -> str:
    """
    Accepts:
      - 'line-productivity'
      - 'line-productivity.csv'
      - '/some/path/line-productivity.csv'
    Returns:
      - 'line-productivity' (stem only)
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return Path(s).stem


def normalize_mapping_inplace(mapping: dict) -> dict:
    """
    Normalize every occurrence of 'source_file' in the mapping to a stem (no .csv, no dirs),
    and normalize metadata.source_files entries to filenames (stem.csv).
    """
    if not isinstance(mapping, dict):
        return mapping

    md = mapping.get("metadata", {})
    if isinstance(md, dict) and "source_files" in md and isinstance(md["source_files"], list):
        normalized = []
        for f in md["source_files"]:
            stem = normalize_source_file(f)
            if stem:
                normalized.append(f"{stem}.csv")
        md["source_files"] = sorted(dict.fromkeys(normalized))
        mapping["metadata"] = md

    def walk(obj):
        if isinstance(obj, dict):
            if "source_file" in obj and obj["source_file"] is not None:
                obj["source_file"] = normalize_source_file(obj["source_file"])
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(mapping)
    return mapping


# ============================================================
# Load + normalize SOURCE tables
# ============================================================

def load_and_normalize_sources_from_mapping(
    mapping_path=OUTPUT_MAPPING_PATH,
    base_path=USER_DATA_DIR,
):
    with open(mapping_path) as f:
        mapping = yaml.safe_load(f) or {}

    mapping = normalize_mapping_inplace(mapping)

    dfs = {}
    for file in mapping["metadata"]["source_files"]:
        path = Path(base_path) / file
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")

        df = pd.read_csv(path, dtype=str)

        # Normalize headers once
        df.columns = df.columns.astype(str).str.strip().str.lower()

        # Drop duplicate column names (keeps first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]

        # Strip all object columns once
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].str.strip()

        dfs[path.stem] = df
        print(f"[LOAD] {file} → {df.shape}")

    return dfs


# ============================================================
# Canonical mapping (run-level) + two-pass fallback joins
# ============================================================

def load_normalize_and_map_to_canonical(
    mapping_path=OUTPUT_MAPPING_PATH,
    base_path=USER_DATA_DIR,
):
    with open(mapping_path) as f:
        mapping = yaml.safe_load(f) or {}

    mapping = normalize_mapping_inplace(mapping)
    source_dfs = load_and_normalize_sources_from_mapping(mapping_path, base_path)

    # --- base run index ---
    run_spec = mapping["run"]["run_id"]
    run_table = normalize_source_file(run_spec["source_file"])
    run_field = run_spec["field"].strip().lower()

    src_df = source_dfs[run_table]
    if run_field not in src_df.columns:
        raise KeyError(
            f"Run field '{run_field}' not found in {run_table}.csv.\n"
            f"Available columns: {list(src_df.columns)}"
        )

    canonical_df = (
        src_df[[run_field]]
        .dropna()
        .drop_duplicates()
        .rename(columns={run_field: "run.run_id"})
        .set_index("run.run_id")
    )
    print(f"[BASE] Built from {run_table}.{run_field}")

    # collect mapped simple fields
    mapped = []
    for section, fields in mapping.items():
        if section in {"run", "stations", "downtime", "metrics", "metadata"}:
            continue
        if not isinstance(fields, dict):
            continue
        for field_name, spec in fields.items():
            if isinstance(spec, dict) and spec.get("unavailable") is True:
                continue
            table = normalize_source_file(spec["source_file"])
            col = spec["field"].strip().lower()
            out = f"{section}.{field_name}"
            mapped.append((table, col, out))

    # --- pass 1: join on run key ---
    deferred = []
    for table, col, out in mapped:
        if table not in source_dfs:
            raise KeyError(f"Mapped source table '{table}' not loaded. Available: {list(source_dfs.keys())}")

        df = source_dfs[table]
        if run_field in df.columns:
            missing = [c for c in (run_field, col) if c not in df.columns]
            if missing:
                raise KeyError(
                    f"Missing columns {missing} in {table}.csv\nAvailable: {list(df.columns)}"
                )
            j = (
                df[[run_field, col]]
                .dropna(subset=[run_field])
                .drop_duplicates(subset=[run_field])
                .set_index(run_field)
                .rename(columns={col: out})
            )
            canonical_df = canonical_df.join(j, how="left")
            print(f"[MAP] {out} ← {table}.{col} (run key)")
        else:
            deferred.append((table, col, out))

    # --- pass 2: fallback join via product.product_code (dimension table like products.csv) ---
    left_key = "product.product_code"
    for table, col, out in deferred:
        df = source_dfs[table]

        if left_key not in canonical_df.columns:
            print(f"[SKIP] {out} ← {table}.{col} (needs {left_key} mapped first)")
            continue

        # try common product-code names on the right side
        candidates = ["product", "product_code", "code", "sku"]
        right_key = next((k for k in candidates if k in df.columns), None)
        if right_key is None:
            print(f"[SKIP] {out} ← {table}.{col} (no product key in {table}.csv; tried {candidates})")
            continue

        if col not in df.columns:
            print(f"[SKIP] {out} ← {table}.{col} (column not found; available: {list(df.columns)})")
            continue

        lookup = (
            df[[right_key, col]]
            .dropna(subset=[right_key])
            .drop_duplicates(subset=[right_key])
            .rename(columns={col: out})
        )

        canonical_df = (
            canonical_df.reset_index()
            .merge(lookup, left_on=left_key, right_on=right_key, how="left")
            .drop(columns=[right_key])
            .set_index("run.run_id")
        )
        print(f"[MAP] {out} ← {table}.{col} (fallback on {left_key} ↔ {table}.{right_key})")

    canonical_df = canonical_df.reset_index()
    if canonical_df.empty:
        raise ValueError("Canonical dataframe is empty")

    return canonical_df


# ============================================================
# Datetime normalization (ALWAYS production_date + extracted time)
# ============================================================

_TIME_RE = re.compile(r"(\d{1,2}:\d{2})(?::(\d{2}))?")

def normalize_datetime_using_production_date(dt_series: pd.Series, production_date_series: pd.Series) -> pd.Series:
    """
    Build datetimes as: production_date + time extracted from dt_series.
    This prevents:
      - 'today' being injected
      - '1900-01-01' or '1900-01-02' garbage dates from time-only parsing
    Accepts dt_series values like:
      - '1/20/2026 16:50'
      - '16:50'
      - '  16:50:00  '
    """
    raw = dt_series.astype(str).str.strip().replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA})

    # production_date → YYYY-MM-DD
    prod_raw = production_date_series.astype(str).str.strip().replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA})
    prod_dt = pd.to_datetime(prod_raw, errors="coerce")
    prod_str = prod_dt.dt.strftime("%Y-%m-%d")

    def extract_time(x):
        if x is pd.NA or x is None:
            return pd.NA
        m = _TIME_RE.search(str(x))
        if not m:
            return pd.NA
        hhmm = m.group(1)
        ss = m.group(2) or "00"
        return f"{hhmm}:{ss}"

    time_str = raw.map(extract_time)

    # Compose datetime strictly from production date + extracted time
    composed = pd.to_datetime(prod_str + " " + time_str.astype("string"), errors="coerce")
    return composed


def enforce_end_after_start_next_day(df: pd.DataFrame, start_col: str, end_col: str) -> None:
    """If end < start, treat end as next day."""
    st = pd.to_datetime(df[start_col], errors="coerce")
    en = pd.to_datetime(df[end_col], errors="coerce")
    rollover = en.notna() & st.notna() & (en < st)
    df.loc[rollover, end_col] = en[rollover] + pd.Timedelta(days=1)


# ============================================================
# Stations (repeated)
# ============================================================

def build_stations_df(source_dfs, stations_mapping):
    rk = stations_mapping["run_key"]
    run_table = normalize_source_file(rk["source_file"])
    run_field = rk["field"].strip().lower()

    rows = []
    for station in stations_mapping.get("stations", []):
        op = station["operator_id"]
        op_table = normalize_source_file(op["source_file"])
        op_field = op["field"].strip().lower()

        df = (
            source_dfs[run_table][[run_field]]
            .merge(
                source_dfs[op_table][[run_field, op_field]],
                on=run_field,
                how="left",
            )
            .rename(columns={run_field: "run_id", op_field: "operator_id"})
        )
        df["station"] = station["station"]
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["run_id", "operator_id", "station"])

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Downtime (repeated)
# ============================================================

def build_downtime_df(source_dfs, downtime_mapping):
    rk = downtime_mapping["run_key"]
    run_field = rk["field"].strip().lower()

    minutes_table = normalize_source_file(downtime_mapping["minutes"]["source_file"])
    factor_spec = downtime_mapping["factor_code"]
    factor_table = normalize_source_file(factor_spec["source_file"])
    factor_field = factor_spec["field"].strip().lower()

    oe_spec = downtime_mapping.get("is_operator_error")
    oe_field = oe_spec["field"].strip().lower() if oe_spec else None

    df_minutes = source_dfs[minutes_table]
    df_factors = source_dfs[factor_table]

    factor_cols = [c for c in df_minutes.columns if c.isdigit()]

    long = df_minutes.melt(
        id_vars=[run_field],
        value_vars=factor_cols,
        var_name="factor_code",
        value_name="minutes",
    )

    long["minutes"] = pd.to_numeric(long["minutes"], errors="coerce")
    long = long.dropna(subset=["minutes"])

    cols = [factor_field] + ([oe_field] if oe_field else [])
    long = (
        long.merge(
            df_factors[cols],
            left_on="factor_code",
            right_on=factor_field,
            how="left",
        )
        .rename(columns={run_field: "run_id"})
        .drop(columns=[factor_field])
    )

    if oe_field:
        long["is_operator_error"] = (
            long[oe_field]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"1", "true", "yes", "y"})
        )
        long = long.drop(columns=[oe_field])
    else:
        long["is_operator_error"] = False

    return long


# ============================================================
# Metrics (derived)
# ============================================================

def build_metrics_df(canonical_df, downtime_df):
    required = {"production.production_date", "timing.start_time", "timing.end_time"}
    missing = required - set(canonical_df.columns)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    df = canonical_df.copy()

    start = pd.to_datetime(df["timing.start_time"], errors="coerce")
    end = pd.to_datetime(df["timing.end_time"], errors="coerce")

    # safety: rollover in case caller forgot
    end = end + pd.to_timedelta((end < start).astype(int), unit="D")

    df["run_minutes"] = (end - start).dt.total_seconds() / 60

    total_dt = downtime_df.groupby("run_id")["minutes"].sum().rename("total_downtime_minutes")
    op_err_dt = downtime_df[downtime_df["is_operator_error"]].groupby("run_id")["minutes"].sum().rename("operator_error_minutes")

    metrics = (
        df[["run.run_id", "run_minutes"]]
        .rename(columns={"run.run_id": "run_id"})
        .merge(total_dt, on="run_id", how="left")
        .merge(op_err_dt, on="run_id", how="left")
        .fillna(0)
    )

    # Optional: overrun_minutes if timing.min_run_minutes exists
    if "timing.min_run_minutes" in df.columns:
        min_run = pd.to_numeric(df["timing.min_run_minutes"], errors="coerce")
        min_run = min_run.fillna(0)
        tmp = df[["run.run_id"]].rename(columns={"run.run_id": "run_id"}).copy()
        tmp["min_run_minutes"] = min_run
        metrics = metrics.merge(tmp, on="run_id", how="left")
        metrics["overrun_minutes"] = (metrics["run_minutes"] - metrics["min_run_minutes"]).clip(lower=0)
        metrics = metrics.drop(columns=["min_run_minutes"])
    else:
        metrics["overrun_minutes"] = 0.0

    # efficiency_pct
    denom = metrics["run_minutes"].replace({0: pd.NA})
    productive = (metrics["run_minutes"] - metrics["total_downtime_minutes"]).clip(lower=0)
    metrics["efficiency_pct"] = (productive / denom) * 100
    metrics["efficiency_pct"] = metrics["efficiency_pct"].fillna(0)

    return metrics


# ============================================================
# Flatten (wide)
# ============================================================

def flatten_for_analysis_wide(runs_df, stations_df, downtime_df, metrics_df):
    runs = runs_df.rename(columns={"run.run_id": "run_id_canonical"})

    stations_wide = (
        stations_df.pivot_table(
            index="run_id",
            columns="station",
            values="operator_id",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"run_id": "run_id_canonical"})
    )

    stations_wide.columns = [
        f"stations.{c.replace(' ', '_').lower()}_operator" if c != "run_id_canonical" else c
        for c in stations_wide.columns
    ]

    downtime_wide = (
        downtime_df.pivot_table(
            index="run_id",
            columns="factor_code",
            values="minutes",
            aggfunc="sum",
        )
        .reset_index()
        .rename(columns={"run_id": "run_id_canonical"})
    )

    downtime_wide.columns = [
        f"downtime.factor_{c}" if c != "run_id_canonical" else c
        for c in downtime_wide.columns
    ]

    metrics = metrics_df.rename(columns={"run_id": "run_id_canonical"})

    return (
        runs
        .merge(stations_wide, on="run_id_canonical", how="left")
        .merge(downtime_wide, on="run_id_canonical", how="left")
        .merge(metrics, on="run_id_canonical", how="left")
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # Build canonical run-level dataframe (includes min_run_minutes via fallback join)
    canonical_df = load_normalize_and_map_to_canonical()
    print("Canonical base dataframe built")

    # Build start/end datetimes STRICTLY from production_date + extracted time
    canonical_df["timing.start_time"] = normalize_datetime_using_production_date(
        canonical_df["timing.start_time"],
        canonical_df["production.production_date"],
    )
    canonical_df["timing.end_time"] = normalize_datetime_using_production_date(
        canonical_df["timing.end_time"],
        canonical_df["production.production_date"],
    )

    # Enforce next-day rollover when end < start
    enforce_end_after_start_next_day(canonical_df, "timing.start_time", "timing.end_time")

    # Load mapping + sources once
    with open(OUTPUT_MAPPING_PATH) as f:
        mapping = yaml.safe_load(f) or {}
    mapping = normalize_mapping_inplace(mapping)

    source_dfs = load_and_normalize_sources_from_mapping()

    # Build repeated sections + metrics + flatten
    stations_df = build_stations_df(source_dfs, mapping["stations"])
    downtime_df = build_downtime_df(source_dfs, mapping["downtime"])
    metrics_df = build_metrics_df(canonical_df, downtime_df)

    flat_df = flatten_for_analysis_wide(
        canonical_df,
        stations_df,
        downtime_df,
        metrics_df,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    flat_df.to_csv(ARTIFACTS_DIR / "canonical_flat.csv", index=False)
    print("Saved canonical_flat.csv")


if __name__ == "__main__":
    main()
