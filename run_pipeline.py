import subprocess
import sys
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
STEPS_DIR = PROJECT_ROOT / "pipeline" / "steps"
ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

PYTHON = sys.executable


# ============================================================
# HELPERS
# ============================================================

def run_step(name: str, script: Path, interactive: bool = False):
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    print(f"\nRunning: {name}")
    print(f"  Script: {script.name}")
    if interactive:
        print("  Mode: INTERACTIVE (user input required)\n")

    result = subprocess.run([PYTHON, str(script)], cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed at step: {name}")

    print(f"Completed: {name}")


def require_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: {path}\n"
            f"Upstream step did not complete successfully or wrote to a different filename."
        )


# ============================================================
# PIPELINE
# ============================================================

def main():
    print("\n==============================")
    print(" OPERATOR SELECTION PIPELINE ")
    print("==============================")

    # --------------------------------------------------------
    # 0. RAW DATA MAPPING (INTERACTIVE)
    # --------------------------------------------------------
    mapping_path = ARTIFACTS_DIR / "metadata_mapping.yml"

    run_mapping = None
    while run_mapping not in {"y", "n"}:
        run_mapping = input(
            "\nRun raw data mapping to (re)generate metadata_mapping.yml? [y/n]: "
        ).strip().lower()

    if run_mapping == "y":
        run_step(
            "Raw data mapping (interactive)",
            STEPS_DIR / "raw_data_mapping.py",
            interactive=True,
        )
        require_artifact(mapping_path)
    else:
        print("\n▶ Skipping raw data mapping")
        print(f"  Using existing: {mapping_path}")
        require_artifact(mapping_path)

    # --------------------------------------------------------
    # 1. DATA INGEST (outputs canonical_flat.csv)
    # --------------------------------------------------------
    run_step(
        "Data ingest & canonical mapping",
        STEPS_DIR / "data_ingest.py",
    )
    require_artifact(ARTIFACTS_DIR / "canonical_flat.csv")

    # --------------------------------------------------------
    # 2. VALIDATION (produces canonical_wide_validated.csv)
    # --------------------------------------------------------
    run_step(
        "Canonical validation (auto)",
        STEPS_DIR / "validate_canonical_auto.py",
    )
    require_artifact(ARTIFACTS_DIR / "canonical_wide_validated.csv")

    # --------------------------------------------------------
    # 3. ANALYSIS (single-input: canonical_wide_validated.csv)
    # --------------------------------------------------------
    run_step(
        "Operator efficiency analysis",
        STEPS_DIR / "operator_efficiency_analysis.py",
    )

    run_step(
        "Operator influence analysis",
        STEPS_DIR / "operator_influence_analysis.py",
    )

    # --------------------------------------------------------
    # 4. ASSIGNMENTS (single-input: canonical_wide_validated.csv)
    # --------------------------------------------------------
    run_step(
        "Operator assignment model (variability)",
        STEPS_DIR / "operator_assignment_model_variability.py",
    )

    print("\n==============================")
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("==============================\n")


if __name__ == "__main__":
    main()
