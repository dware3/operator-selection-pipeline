# app/main.py
from __future__ import annotations

import json
import hashlib
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse

# Matplotlib (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="Operator Selection Reports API")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Operator Selection Report</title>
</head>
<body style="font-family:system-ui; padding:40px">
  <h1>Operator Selection Pipeline</h1>

  <p>Generate the latest report and open it.</p>

  <!-- OPTION A: same-tab navigation -->
  <form action="/report/build-and-open" method="post">
    <button type="submit" style="padding:12px 16px; font-size:14px;">
      Generate & Open Report
    </button>
  </form>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Paths (repo layout assumes: <root>/app/main.py and <root>/pipeline/artifacts)
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

REPORT_DIR = ARTIFACTS_DIR / "report"
FIG_DIR = REPORT_DIR / "figures"
EXAMPLE_DIR = REPORT_DIR / "examples"
MANIFEST_PATH = REPORT_DIR / "report_manifest.json"

# Core pipeline artifacts
CANONICAL_VALIDATED = ARTIFACTS_DIR / "canonical_wide_validated.csv"
DOWNTIME_VARIABILITY = ARTIFACTS_DIR / "operator_downtime_variability.csv"
FACTOR_TESTS = ARTIFACTS_DIR / "operator_factor_tests.csv"
PRODUCT_DOWNTIME = ARTIFACTS_DIR / "operator_product_downtime.csv"
ASSIGNMENTS = ARTIFACTS_DIR / "product.product_code_operator_assignments.csv"


# -----------------------------------------------------------------------------
# Minimal utilities (kept because build + file-serving need them)
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    # Protect file-serving routes used by the HTML report <img src="...">
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return name


def _guess_media_type(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".svg"):
        return "image/svg+xml"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".json"):
        return "application/json"
    if fn.endswith((".yml", ".yaml")):
        return "text/yaml"
    if fn.endswith(".csv"):
        return "text/csv"
    return "application/octet-stream"


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found: {path}")


def _file_fingerprint(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    # Used only for the manifest (kept because your report prints inputs_pretty)
    if path is None or not path.exists():
        return None
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))  # first 1MB
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "sha256_1mb": h.hexdigest(),
    }


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    _require(path, label)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    return None


def _pick_numeric_value_col(df: pd.DataFrame, preferred: List[str]) -> str:
    c = _pick_first_existing(df.columns.tolist(), preferred)
    if c:
        return c

    numeric_cols = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(col)

    if not numeric_cols:
        raise HTTPException(status_code=400, detail="No numeric column found to plot in this artifact.")
    return numeric_cols[0]


def _watermark(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=36, alpha=0.12, rotation=25)


def _save_fig(fig: plt.Figure, out_path: Path, watermark_text: Optional[str] = None) -> None:
    _ensure_dir(out_path.parent)
    if watermark_text:
        _watermark(fig, watermark_text)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Column pickers for canonical_wide_validated
# -----------------------------------------------------------------------------
def _load_validated_df() -> pd.DataFrame:
    return _read_csv(CANONICAL_VALIDATED, "Validated canonical dataset (canonical_wide_validated.csv)")


def _pick_operator_col(df: pd.DataFrame) -> str:
    cols = df.columns.astype(str).tolist()

    exact_candidates = [
        "operator",
        "operator_id",
        "stations.operator_id",
        "station.operator_id",
        "stations.operator",
        "station.operator",
    ]
    for c in exact_candidates:
        if c in df.columns:
            return c

    lowered = [(c, c.lower()) for c in cols]
    preferred = [
        c for c, lc in lowered
        if ("operator" in lc) and ("assigned" not in lc) and ("assignment" not in lc)
    ]
    if preferred:
        station_pref = [c for c in preferred if "station" in c.lower() or "stations" in c.lower()]
        return station_pref[0] if station_pref else preferred[0]

    fallback = [c for c, lc in lowered if lc in {"op", "ops"} or lc.startswith("op_") or lc.endswith("_op")]
    if fallback:
        return fallback[0]

    raise HTTPException(
        status_code=400,
        detail=(
            "Could not find operator column. "
            "Try inspecting canonical_wide_validated.csv headers and add it to exact_candidates. "
            f"Available columns include: {cols[:60]}"
        ),
    )


def _pick_efficiency_col(df: pd.DataFrame) -> str:
    cols = df.columns.astype(str).tolist()

    exact_candidates = [
        "efficiency",
        "efficiency_pct",
        "metrics.efficiency",
        "metrics.efficiency_pct",
    ]
    for c in exact_candidates:
        if c in df.columns:
            return c

    lowered = [(c, c.lower()) for c in cols]
    preferred = [c for c, lc in lowered if "efficiency" in lc]
    if preferred:
        return preferred[0]

    alt = [c for c, lc in lowered if lc in {"outcome", "performance"} or "outcome" in lc or "performance" in lc]
    if alt:
        return alt[0]

    raise HTTPException(
        status_code=400,
        detail=(
            "Could not find efficiency column. "
            "Try inspecting canonical_wide_validated.csv headers and add it to exact_candidates. "
            f"Available columns include: {cols[:60]}"
        ),
    )


# -----------------------------------------------------------------------------
# Figure builders (REAL)
# -----------------------------------------------------------------------------
def fig_efficiency_by_operator(df: pd.DataFrame) -> plt.Figure:
    op_col = _pick_operator_col(df)
    eff_col = _pick_efficiency_col(df)

    clean = df[[op_col, eff_col]].dropna()
    y = pd.to_numeric(clean[eff_col], errors="coerce").dropna()

    if y.empty:
        raise HTTPException(status_code=400, detail="No efficiency values found after cleaning.")

    # Normalize to 0–1 if it looks like percent
    if y.max() > 1.5:
        clean = clean.assign(_y=pd.to_numeric(clean[eff_col], errors="coerce") / 100.0)
        ylab = "Efficiency (0–1)"
    else:
        clean = clean.assign(_y=pd.to_numeric(clean[eff_col], errors="coerce"))
        ylab = "Efficiency (0–1)"

    groups = []
    labels = []
    for op, g in clean.groupby(op_col):
        vals = g["_y"].dropna().astype(float).to_numpy()
        if len(vals):
            groups.append(vals)
            labels.append(str(op))

    if not groups:
        raise HTTPException(status_code=400, detail="No operator/efficiency pairs after cleaning.")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(groups, labels=labels, showfliers=True)
    ax.set_title("Efficiency by Operator")
    ax.set_xlabel("Operator")
    ax.set_ylabel(ylab)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def fig_run_counts_by_operator(df: pd.DataFrame) -> plt.Figure:
    op_col = _pick_operator_col(df)
    counts = df[op_col].dropna().astype(str).value_counts().sort_index()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(counts.index.tolist(), counts.values.tolist())
    ax.set_title("Run Counts by Operator (Sample Size)")
    ax.set_xlabel("Operator")
    ax.set_ylabel("Runs")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def fig_operator_variability_bar(var_df: pd.DataFrame) -> plt.Figure:
    cols = var_df.columns.astype(str).tolist()
    op_col = "operator" if "operator" in cols else cols[0]
    std_col = "std" if "std" in cols else _pick_numeric_value_col(var_df, preferred=["std_downtime_rate", "std", "stdev"])
    runs_col = "runs" if "runs" in cols else None

    df = var_df[[op_col, std_col] + ([runs_col] if runs_col in cols else [])].copy()
    df[op_col] = df[op_col].astype(str)
    df[std_col] = pd.to_numeric(df[std_col], errors="coerce")
    df = df.dropna(subset=[std_col]).sort_values(std_col, ascending=True)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(df[op_col].tolist(), df[std_col].to_numpy(dtype=float))
    ax.set_title("Downtime-Rate Variability by Operator (Std Dev)")
    ax.set_xlabel("Operator")
    ax.set_ylabel(std_col)
    ax.grid(True, axis="y", alpha=0.3)

    if runs_col in df.columns:
        for i, (op, stdv, runs) in enumerate(zip(df[op_col], df[std_col], df[runs_col])):
            ax.text(i, float(stdv), f" n={int(runs)}", ha="center", va="bottom", fontsize=8)

    return fig


def fig_factor_tests(artifact: pd.DataFrame) -> plt.Figure:
    cols = artifact.columns.tolist()
    label_col = _pick_first_existing(cols, ["factor", "test", "downtime_factor", "factor_name", "name"])
    p_col = _pick_first_existing(cols, ["p_value", "pvalue", "p", "prob", "probability"])

    if not label_col:
        for c in cols:
            if pd.to_numeric(artifact[c], errors="coerce").isna().sum() > 0:
                label_col = c
                break

    if not label_col:
        raise HTTPException(status_code=400, detail=f"Could not identify a label column in factor tests. Columns: {cols}")

    if p_col:
        tmp = artifact[[label_col, p_col]].copy()
        tmp[p_col] = pd.to_numeric(tmp[p_col], errors="coerce")
        tmp = tmp.dropna(subset=[p_col]).sort_values(p_col, ascending=True).head(20)
        if tmp.empty:
            raise HTTPException(status_code=400, detail="No usable p-values found in factor tests.")
        vals = -np.log10(np.clip(tmp[p_col].to_numpy(dtype=float), 1e-12, 1.0))
        title = "Operator Factor Tests (-log10 p-value) (Top 20 most significant)"
        xlab = "-log10(p)"
    else:
        val_col = _pick_numeric_value_col(artifact, preferred=["statistic", "f_statistic", "score"])
        tmp = artifact[[label_col, val_col]].copy()
        tmp[val_col] = pd.to_numeric(tmp[val_col], errors="coerce")
        tmp = tmp.dropna(subset=[val_col]).sort_values(val_col, ascending=False).head(20)
        vals = tmp[val_col].to_numpy(dtype=float)
        title = f"Operator Factor Tests ({val_col}) (Top 20)"
        xlab = val_col

    labels = tmp[label_col].astype(str).tolist()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.barh(labels[::-1], vals[::-1])
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.grid(True, axis="x", alpha=0.3)

    if p_col:
        ax.axvline(-np.log10(0.05), linestyle="--", linewidth=1)
        ax.text(-np.log10(0.05), 0, " α=0.05", va="bottom", ha="left")

    return fig


def fig_assignment_summary_table(assign_df: pd.DataFrame) -> plt.Figure:
    a = assign_df.copy()
    a.columns = [str(c).strip() for c in a.columns]

    key_col = a.columns[0]
    preferred = ["assigned_operator", "confidence", "runs", "mean_downtime_rate", "std_downtime_rate", "explanation"]
    cols = [c for c in preferred if c in a.columns]
    if not cols:
        cols = a.columns.tolist()[1:7]

    view = a[[key_col] + cols].head(12).copy()

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    tbl = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    ax.set_title("Assignment Summary (Top Rows)")
    return fig


# -----------------------------------------------------------------------------
# Example (SIMULATED) figures
# -----------------------------------------------------------------------------
def simulate_strong_operator_effect(df: pd.DataFrame, effect_size: float = 0.15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sim = df.copy()

    op_col = _pick_operator_col(sim)
    eff_col = _pick_efficiency_col(sim)

    base = pd.to_numeric(sim[eff_col], errors="coerce")
    if base.max(skipna=True) > 1.5:
        base = base / 100.0

    baseline = pd.Series(0.80 + rng.normal(0, 0.03, size=len(sim)), index=sim.index)
    base = base.fillna(baseline)

    ops = sim[op_col].astype(str).fillna("unknown")
    unique_ops = sorted(ops.unique().tolist())

    if len(unique_ops) == 1:
        shifts = {unique_ops[0]: effect_size}
    else:
        span = np.linspace(-effect_size / 2, effect_size / 2, num=len(unique_ops))
        shifts = {op: float(s) for op, s in zip(unique_ops, span)}

    shift_vec = ops.map(shifts).astype(float).to_numpy()
    noise = rng.normal(0, 0.02, size=len(sim))

    sim_eff = np.clip(base.to_numpy(dtype=float) + shift_vec + noise, 0.0, 1.0)
    sim[eff_col] = sim_eff
    return sim


def simulate_strong_downtime_variability_effect(
    var_df: pd.DataFrame,
    spread: float = 0.14,
    floor: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    cols = var_df.columns.astype(str).tolist()
    op_col = "operator" if "operator" in cols else cols[0]
    std_col = "std" if "std" in cols else _pick_numeric_value_col(var_df, preferred=["std_downtime_rate", "std", "stdev"])
    runs_col = "runs" if "runs" in cols else None

    base = var_df.copy()
    base[op_col] = base[op_col].astype(str)

    keep_cols = [op_col]
    if runs_col in cols:
        keep_cols.append(runs_col)
    base = base[keep_cols].dropna(subset=[op_col])

    ops = sorted(base[op_col].unique().tolist())
    n_ops = max(len(ops), 1)

    target = np.linspace(floor, floor + spread, num=n_ops)
    noise = rng.normal(0, spread * 0.03, size=n_ops)
    std_vals = np.clip(target + noise, 0.0, None)

    sim = pd.DataFrame({op_col: ops, std_col: std_vals})

    if runs_col in base.columns:
        runs_map = base.drop_duplicates(subset=[op_col]).set_index(op_col)[runs_col]
        sim[runs_col] = pd.to_numeric(sim[op_col].map(runs_map), errors="coerce").fillna(8).astype(int)
    else:
        sim["runs"] = rng.integers(low=6, high=20, size=n_ops)
        runs_col = "runs"

    if std_col != "std":
        sim = sim.rename(columns={std_col: "std"})
    if runs_col != "runs":
        sim = sim.rename(columns={runs_col: "runs"})
    sim = sim.rename(columns={op_col: "operator"})
    return sim


def simulate_strong_factor_test_signal(ft_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [str(c).strip() for c in ft_df.columns.tolist()]

    label_col = _pick_first_existing(cols, ["factor", "test", "downtime_factor", "factor_name", "name"])
    p_col = _pick_first_existing(cols, ["p_value", "pvalue", "p", "prob", "probability"])

    if not label_col:
        for c in cols:
            if pd.to_numeric(ft_df[c], errors="coerce").isna().sum() > 0:
                label_col = c
                break
    if not label_col:
        label_col = cols[0]

    labels = ft_df[label_col].astype(str).fillna("").tolist()
    labels = [x for x in labels if x.strip()]
    if not labels:
        labels = ["Factor A", "Factor B", "Factor C", "Factor D", "Factor E", "Factor F"]

    labels = labels[:20] if len(labels) > 20 else labels
    n = len(labels)

    sim = pd.DataFrame({label_col: labels})

    if p_col:
        k = max(3, n // 2)
        strong = np.geomspace(1e-10, 1e-4, num=k)
        weak = rng.uniform(0.08, 0.8, size=n - k)
        pvals = np.concatenate([strong, weak])
        rng.shuffle(pvals)
        sim[p_col] = pvals
    else:
        val_col = _pick_numeric_value_col(ft_df, preferred=["statistic", "f_statistic", "score"])
        k = max(3, n // 2)
        strong = rng.uniform(8.0, 20.0, size=k)
        weak = rng.uniform(0.2, 5.0, size=n - k)
        stats = np.concatenate([strong, weak])
        rng.shuffle(stats)
        sim[val_col] = stats

    return sim


def fig_example_efficiency_separation(df: pd.DataFrame) -> plt.Figure:
    sim = simulate_strong_operator_effect(df)
    fig = fig_efficiency_by_operator(sim)
    fig.axes[0].set_title("SIMULATED EXAMPLE: Clear Efficiency Differences by Operator")
    return fig


def fig_example_downtime_variability_separation(var_df: pd.DataFrame) -> plt.Figure:
    sim = simulate_strong_downtime_variability_effect(var_df)
    fig = fig_operator_variability_bar(sim)
    fig.axes[0].set_title("SIMULATED EXAMPLE: Clear Separation in Downtime-Rate Variability (Std Dev)")
    return fig


def fig_example_factor_tests_strong_signal(ft_df: pd.DataFrame) -> plt.Figure:
    sim = simulate_strong_factor_test_signal(ft_df)
    fig = fig_factor_tests(sim)
    fig.axes[0].set_title("SIMULATED EXAMPLE: Strong Operator Signal in Factor Tests")
    return fig


# -----------------------------------------------------------------------------
# Simplified UI: ONE button → build → open report
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        content="""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Operator Selection Report</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin:0; background:#fff; color:#111; }
    .wrap { max-width: 720px; margin: 0 auto; padding: 48px 18px; }
    .card { background:#f7f7f8; border:1px solid #e6e6e9; border-radius: 12px; padding: 20px; }
    h1 { margin:0 0 10px; font-size: 22px; }
    p { margin:8px 0 16px; color:#555; line-height:1.45; }
    button { border:0; background:#0b57d0; color:#fff; border-radius:10px; padding:12px 16px; font-size:14px; cursor:pointer; }
    button:hover { filter: brightness(0.95); }
    .small { font-size:12px; color:#666; margin-top:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Operator Selection Pipeline</h1>
      <p>One click: build the latest report and open the HTML report.</p>
      <form action="/report/build-and-open" method="post" target="_blank">
        <button type="submit">Generate &amp; Open Report</button>
      </form>
      <div class="small">If nothing opens, allow pop-ups for localhost and retry.</div>
    </div>
  </div>
</body>
</html>
""",
        status_code=200,
    )


@app.post("/report/build-and-open")
def report_build_and_open():
    build_report()
    return RedirectResponse(url="/report/html", status_code=303)


# -----------------------------------------------------------------------------
# Build report (kept; your report depends on its outputs + manifest)
# -----------------------------------------------------------------------------
@app.post("/report/build")
def build_report():
    _ensure_dir(REPORT_DIR)
    _ensure_dir(FIG_DIR)
    _ensure_dir(EXAMPLE_DIR)

    wide = _load_validated_df()
    var_df = _read_csv(DOWNTIME_VARIABILITY, "operator_downtime_variability.csv")
    ft_df = _read_csv(FACTOR_TESTS, "operator_factor_tests.csv")
    pod_df = _read_csv(PRODUCT_DOWNTIME, "operator_product_downtime.csv")
    asn_df = _read_csv(ASSIGNMENTS, "product.product_code_operator_assignments.csv")

    outputs: list[str] = []
    example_outputs: list[str] = []

    p = FIG_DIR / "efficiency_by_operator.png"
    _save_fig(fig_efficiency_by_operator(wide), p)
    outputs.append(p.name)

    p = FIG_DIR / "run_counts_by_operator.png"
    _save_fig(fig_run_counts_by_operator(wide), p)
    outputs.append(p.name)

    p = FIG_DIR / "downtime_variability_by_operator.png"
    _save_fig(fig_operator_variability_bar(var_df), p)
    outputs.append(p.name)

    p = FIG_DIR / "operator_factor_tests.png"
    _save_fig(fig_factor_tests(ft_df), p)
    outputs.append(p.name)

    # NOTE: This reuses the same bar builder; keep if your artifact is already 1D.
    p = FIG_DIR / "product_operator_downtime.png"
    _save_fig(fig_operator_variability_bar(pod_df), p)
    outputs.append(p.name)

    p = FIG_DIR / "product_operator_assignment_summary.png"
    _save_fig(fig_assignment_summary_table(asn_df), p)
    outputs.append(p.name)

    ex = EXAMPLE_DIR / "example_clear_efficiency_by_operator.png"
    _save_fig(fig_example_efficiency_separation(wide), ex, watermark_text="SIMULATED EXAMPLE")
    example_outputs.append(ex.name)

    ex = EXAMPLE_DIR / "example_clear_downtime_variability_by_operator.png"
    _save_fig(fig_example_downtime_variability_separation(var_df), ex, watermark_text="SIMULATED EXAMPLE")
    example_outputs.append(ex.name)

    ex = EXAMPLE_DIR / "example_clear_operator_factor_tests.png"
    _save_fig(fig_example_factor_tests_strong_signal(ft_df), ex, watermark_text="SIMULATED EXAMPLE")
    example_outputs.append(ex.name)

    manifest = {
        "generated_at_utc": _utc_now_iso(),
        "inputs": {
            "canonical_wide_validated": _file_fingerprint(CANONICAL_VALIDATED),
            "operator_downtime_variability": _file_fingerprint(DOWNTIME_VARIABILITY),
            "operator_factor_tests": _file_fingerprint(FACTOR_TESTS),
            "operator_product_downtime": _file_fingerprint(PRODUCT_DOWNTIME),
            "product_product_code_operator_assignments": _file_fingerprint(ASSIGNMENTS),
        },
        "outputs": {
            "figures_dir": str(FIG_DIR),
            "examples_dir": str(EXAMPLE_DIR),
            "figures": outputs,
            "examples": example_outputs,
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"status": "ok", **manifest}


# -----------------------------------------------------------------------------
# File serving for the report's <img src="/report/..."> links
# (Kept: required for HTML report to display images.)
# -----------------------------------------------------------------------------
@app.get("/report/figures/{filename}")
def get_report_figure(filename: str):
    filename = _safe_filename(filename)
    path = FIG_DIR / filename
    _require(path, "Report figure")
    return FileResponse(path, media_type=_guess_media_type(filename), filename=filename)


@app.get("/report/examples/{filename}")
def get_example_figure(filename: str):
    filename = _safe_filename(filename)
    path = EXAMPLE_DIR / filename
    _require(path, "Example figure")
    return FileResponse(path, media_type=_guess_media_type(filename), filename=filename)


# -----------------------------------------------------------------------------
# HTML report 
# -----------------------------------------------------------------------------

REPORT_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Operator Selection Pipeline Report</title>

  <style>
    :root {{
      --fg:#111;
      --muted:#555;
      --bg:#fff;
      --card:#f7f7f8;
      --border:#e6e6e9;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}

    body {{
      font-family: var(--sans);
      color: var(--fg);
      background: var(--bg);
      margin: 0;
    }}

    .wrap {{
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 18px 64px;
    }}

    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    h2 {{ margin: 26px 0 10px; font-size: 20px; }}
    h3 {{ margin: 18px 0 8px; font-size: 16px; }}

    p {{
      margin: 8px 0;
      line-height: 1.45;
    }}

    .muted {{ color: var(--muted); }}

    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 14px;
      margin: 12px 0;
    }}

    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}

    @media (min-width: 860px) {{
      .grid.two {{
        grid-template-columns: 1fr 1fr;
      }}
    }}

    img {{
      width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
    }}

    .caption {{
      font-size: 13px;
      color: var(--muted);
      margin-top: 8px;
      line-height: 1.45;
    }}

    code, pre {{
      font-family: var(--mono);
      font-size: 12px;
    }}

    pre {{
      background: #0b0f14;
      color: #e6edf3;
      padding: 10px 12px;
      border-radius: 10px;
      overflow: auto;
      border: 1px solid #1b2430;
    }}

    .pill {{
      display: inline-block;
      font-size: 12px;
      padding: 3px 9px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      margin-right: 8px;
    }}

    a {{
      color: #0b57d0;
      text-decoration: none;
    }}

    a:hover {{ text-decoration: underline; }}

    .hr {{
      height: 1px;
      background: var(--border);
      margin: 18px 0;
    }}

    .small {{ font-size: 12px; }}

    /* --- Added: side-by-side pair for observed vs simulated --- */
    .pair {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      margin-top: 8px;
    }}

    @media (min-width: 860px) {{
      .pair {{
        grid-template-columns: 1fr 1fr;
      }}
    }}

    .panel {{
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }}

    .panel-title {{
      font-size: 12px;
      color: var(--muted);
      margin: 0 0 8px;
      text-transform: uppercase;
      letter-spacing: .2px;
    }}

    .wm {{ position: relative; }}

    .wm::after {{
      content: "SIMULATED (SYNTHETIC)";
      position: absolute;
      right: 10px;
      top: 10px;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(255,255,255,.85);
      border: 1px solid var(--border);
      color: var(--muted);
    }}
  </style>
</head>

<body>
  <div class="wrap">
    <h1>Operator Selection Pipeline Report</h1>

    <div class="muted small">
      <span class="pill">Generated: {generated_at}</span>
      <span class="pill">Artifacts: pipeline/artifacts/</span>
    </div>

    <div class="card">
      <h2 style="margin-top:0;">Executive summary</h2>
      <p>
        This report summarizes operator performance signals (efficiency and downtime behavior) and provides a conservative
        assignment recommendation based on observed variability. Where the data does not support strong conclusions,
        the report explicitly indicates low confidence rather than over-claiming. "Simulated example” visuals are synthetic
        and watermarked; they show what a clear association would look like, not what was observed.
      </p>
    </div>

    <h2>Assignments</h2>

    <div class="card">
      <h3 style="margin-top:0;">Recommended assignments (summary)</h3>
      <img src="/report/figures/product_operator_assignment_summary.png" alt="Assignment summary" />
      <div class="caption">
        Interpretation: Assignments are based on observed variability; “low/very low” confidence usually means few runs.
      </div>

      <p class="small muted" style="margin-top:10px;">
        Want the raw table? Download:
        <a href="/artifacts/product.product_code_operator_assignments.csv">
          product.product_code_operator_assignments.csv
        </a>
      </p>
    </div>

    <!-- OBSERVED RESULTS -->
    <h2>Observed results</h2>

    <!-- FIRST GRAPH: DOWNTIME VARIABILITY, WITH SIMULATED EXAMPLE TO THE RIGHT -->
    <div class="card">
      <h3 style="margin-top:0;">Downtime-rate variability by operator</h3>

      <div class="pair">
        <div class="panel">
          <div class="panel-title">Observed</div>
          <img
            src="/report/figures/downtime_variability_by_operator.png"
            alt="Downtime-rate variability by operator (observed)"
          />
        </div>

        <div class="panel wm">
          <div class="panel-title">Simulated example</div>
          <img
            src="/report/examples/example_clear_downtime_variability_by_operator.png"
            alt="Simulated example: strong variability association"
          />
        </div>
      </div>

      <div class="caption">
        <b>About this bar chart:</b> Downtime-rate variability by operator chart shows how consistent each operator’s downtime performance is across runs, using the standard deviation of the downtime rate as the measure of variability. For each run, downtime rate is calculated as downtime minutes divided by total run minutes, and the spread of those per-run rates is summarized for each operator. Lower bars indicate more predictable, stable downtime behavior, while higher bars indicate greater volatility and operational risk, even if average downtime is similar. The n = # label above each bar shows how many runs the calculation is based on and should be used to judge confidence, since variability estimates from few runs are unreliable.
      </div>
    </div>
    
    <!-- EFFICIENCY BY OPERATOR -->
    <div class="card">
      <h3 style="margin-top:0;">Efficiency by operator</h3>

      <div class="pair">
        <div class="panel">
          <div class="panel-title">Observed</div>
          <img
            src="/report/figures/efficiency_by_operator.png"
            alt="Efficiency by operator (observed)"
          />
        </div>

        <div class="panel wm">
          <div class="panel-title">Simulated example</div>
          <img
            src="/report/examples/example_clear_efficiency_by_operator.png"
            alt="Simulated example: clear efficiency separation"
          />
        </div>
      </div>

      <div class="caption">
        <b>About this box-and-whisker plot:</b> The box-and-whisker plot displays a box around the middle 50% of values (the interquartile range), with the median represented by the line inside the box. The whiskers extend to show the range of typical extreme values outside the box. The Efficiency by operator chart compares the distribution of per-run efficiency values across operators, where efficiency is a normalized ratio from 0 to 1 representing the proportion of scheduled run time spent producing output. Each operator’s distribution shows typical performance (the median) as well as run-to-run spread, allowing you to assess both level and consistency. Higher medians indicate more efficient typical runs, while wider boxes or longer whiskers indicate greater variability. When operator distributions overlap heavily, the data does not support strong differentiation and conclusions should be treated as low confidence, especially when run counts are small.
      </div>
    </div>

    <!-- OPERATOR FACTOR TESTS -->
    <div class="card">
      <h3 style="margin-top:0;">Operator factor tests</h3>

      <div class="pair">
        <div class="panel">
          <div class="panel-title">Observed</div>
          <img
            src="/report/figures/operator_factor_tests.png"
            alt="Efficiency by operator (observed)"
          />
        </div>

        <div class="panel wm">
          <div class="panel-title">Simulated example</div>
          <img
            src="/report/examples/example_clear_operator_factor_tests.png"
            alt="Simulated example: strong operator factor-test signal"
          />
        </div>
      </div>

      <div class="caption">
        <b>About this horizontal bar chart:</b> The Operator factor tests chart summarizes statistical tests that evaluate whether operator identity explains meaningful variation in specific downtime factors. Each downtime factor is tested separately to assess whether differences between operators are stronger than would be expected by chance, with larger test statistics or smaller p-values indicating stronger evidence. When a factor shows stronger evidence, targeted training on that specific issue is a reasonable intervention to test. Results should be interpreted as evidence strength only and considered alongside run counts, distribution plots, and operational context, especially when data are limited or uneven.
      </div>
    </div>
    
    <h2>Data & method</h2>

    <div class="card">
      <p>
        Inputs are derived from the validated canonical dataset and downstream analysis artifacts produced by the CLI pipeline.
        Visuals below are generated read-only from those artifacts.
      </p>

      <div class="hr"></div>

      <p class="small muted" style="margin-bottom:6px;">
        Reproducibility (input fingerprints)
      </p>

      <pre>{inputs_pretty}</pre>
    </div>

    <h2>Limitations</h2>

    <div class="card">
      <ul style="margin:8px 0 0 18px; line-height:1.45;">
        <li>Small sample sizes and uneven operator coverage can hide real effects or create noisy false patterns.</li>
        <li>Assignments are risk-minimizing given observed variability; they are not causal claims.</li>
        <li>Additional runs per operator/product/station improve confidence more than more complex modeling.</li>
      </ul>
    </div>

    <div class="muted small" style="margin-top:18px;">
      Generated by the Reports API. If figures are missing, run <code>POST /report/build</code> first.
    </div>
  </div>
</body>
</html>
"""


@app.get("/report/html", response_class=HTMLResponse)
def report_html():
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=400, detail="No report manifest yet. POST /report/build first.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    generated_at = html.escape(str(manifest.get("generated_at_utc", "")))
    inputs_pretty = html.escape(json.dumps(manifest.get("inputs", {}), indent=2, sort_keys=True))

    html_doc = REPORT_HTML_TEMPLATE.format(
        generated_at=generated_at,
        inputs_pretty=inputs_pretty,
    )
    return HTMLResponse(content=html_doc, status_code=200)

@app.post("/report/build-and-open")
def report_build_and_open():
    build_report()  # existing function — do not rewrite
    return RedirectResponse(url="/report/html", status_code=303)
