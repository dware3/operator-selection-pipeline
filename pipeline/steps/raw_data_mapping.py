import pandas as pd
import yaml
from pathlib import Path

# ============================================================
# PATH SETUP (MATCHING SCRIPT B)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
USER_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "pipeline" / "artifacts"

CANONICAL_SCHEMA_PATH = CONFIGS_DIR / "canonical_schema.yml"
OUTPUT_MAPPING_PATH = ARTIFACTS_DIR / "metadata_mapping.yml"

# ============================================================
# UTILITIES
# ============================================================

def prompt_for_file(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def choose_file_from_directory(prompt, directory, suffix):
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob(f"*{suffix}"))

    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {directory}")

    print(f"\n{prompt}")
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}")

    choice = input("> ").strip()
    if choice == "":
        raise ValueError("Selection required")

    return files[int(choice)]


# ============================================================
# SCHEMA PARSING
# ============================================================

def parse_canonical_schema(canonical_schema):
    SKIPPED_SECTIONS = {"stations", "downtime", "metrics", "metadata"}
    parsed = {}
    skipped = []

    for section_name, section_body in canonical_schema.items():
        if section_name in SKIPPED_SECTIONS:
            skipped.append(section_name)
            continue

        if not isinstance(section_body, dict):
            continue

        fields = []
        for field_name, field_def in section_body.items():
            if field_name in {"type", "description"}:
                continue
            if not isinstance(field_def, dict):
                field_def = {"type": field_def}
            fields.append({
                "canonical_field": f"{section_name}.{field_name}",
                "field_name": field_name,
                "definition": field_def,
            })

        if fields:
            parsed[section_name] = fields

    return parsed, skipped


# ============================================================
# METADATA → CANONICAL MAPPING
# ============================================================

def map_metadata_column_to_canonical_field(metadata_df, parsed_schema):
    mapping = {}
    for section_name, fields in parsed_schema.items():
        print(f"\n=== Mapping section: {section_name} ===\n")
        section_block = {}

        for field in fields:
            canonical_path = field["canonical_field"]
            _, field_name = canonical_path.split(".", 1)

            print(metadata_df[["Table", "Field"]].reset_index(drop=True))
            print(f"\nMapping canonical field: {canonical_path}")
            print("Enter row number to map, or press Enter to skip.")

            choice = input("> ").strip()
            if not choice:
                continue

            row = metadata_df.iloc[int(choice)]

            section_block[field_name] = {
                "source_file": f"{row['Table']}.csv",
                "field": row["Field"],
            }

        if section_block:
            mapping[section_name] = section_block

    return mapping


# ============================================================
# STATIONS (REPEATED)
# ============================================================

def map_station_section(metadata_df):
    print("\nMapping canonical section: stations")
    selectable = metadata_df.dropna(subset=["Table", "Field"]).reset_index(drop=True)

    count = int(input("Enter number of stations per run: ").strip())

    print("\nSelect run key field:")
    print(selectable[["Table", "Field"]])
    rk_row = selectable.iloc[int(input("> ").strip())]

    run_key = {
        "source_file": f"{rk_row['Table']}.csv",
        "field": rk_row["Field"],
    }

    stations = []

    for i in range(1, count + 1):
        print(f"\n--- Station {i} ---")
        print("Select operator field (Enter to skip station):")
        print(selectable[["Table", "Field"]])

        choice = input("> ").strip()
        if not choice:
            continue

        row = selectable.iloc[int(choice)]

        stations.append({
            "station": f"Station {i}",
            "operator_id": {
                "source_file": f"{row['Table']}.csv",
                "field": row["Field"],
            },
        })

    if not stations:
        return None

    return {
        "type": "repeated",
        "run_key": run_key,
        "stations": stations,
    }


# ============================================================
# DOWNTIME (REPEATED)
# ============================================================

def map_downtime(metadata_df):
    print("\nMapping canonical section: downtime")
    selectable = metadata_df.dropna(subset=["Table", "Field"]).reset_index(drop=True)

    print("\nSelect run key field:")
    print(selectable[["Table", "Field"]])
    rk_row = selectable.iloc[int(input("> ").strip())]

    run_key = {
        "source_file": f"{rk_row['Table']}.csv",
        "field": rk_row["Field"],
    }

    tables = sorted(metadata_df["Table"].dropna().unique())

    print("\nSelect downtime minutes table:")
    for i, t in enumerate(tables):
        print(f"  [{i}] {t}")
    minutes_table = tables[int(input("> ").strip())]

    print("\nSelect factor-code field:")
    print(selectable[["Table", "Field"]])
    fc_row = selectable.iloc[int(input("> ").strip())]

    print("\nSelect operator_error flag field (Enter to skip):")
    print(selectable[["Table", "Field"]])
    oe_choice = input("> ").strip()

    operator_error = None
    if oe_choice:
        oe_row = selectable.iloc[int(oe_choice)]
        operator_error = {
            "source_file": f"{oe_row['Table']}.csv",
            "field": oe_row["Field"],
        }

    return {
        "type": "repeated",
        "run_key": run_key,
        "factor_code": {
            "source_file": f"{fc_row['Table']}.csv",
            "field": fc_row["Field"],
        },
        "minutes": {
            "source_file": f"{minutes_table}.csv",
            "derive_from": "column_value",
        },
        "is_operator_error": operator_error,
    }


# ============================================================
# METRICS (DERIVED ONLY)
# ============================================================

def declare_metrics_schema_mapping():
    return {
        "metrics": {
            "run_minutes": {
                "derived": True,
                "source": "timing.end_time - timing.start_time",
            },
            "total_downtime_minutes": {
                "derived": True,
                "source": "sum(downtime.minutes)",
            },
            "operator_error_minutes": {
                "derived": True,
                "source": "sum(downtime.minutes where is_operator_error = true)",
            },
        }
    }


# ============================================================
# CONTEXT LOADING
# ============================================================

def load_mapping_context():
    print(f"\nUsing user data directory: {USER_DATA_DIR}")

    metadata_path = choose_file_from_directory(
        prompt="Select metadata file:",
        directory=USER_DATA_DIR,
        suffix=".csv",
    )

    if not CANONICAL_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing canonical schema: {CANONICAL_SCHEMA_PATH}")

    metadata_df = pd.read_csv(metadata_path, dtype=str)

    metadata_df["Field"] = (
        metadata_df["Field"]
        .astype("string")
        .str.strip()
        .replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA})
    )

    metadata_df = metadata_df.dropna(subset=["Field"])

    with open(CANONICAL_SCHEMA_PATH, "r") as f:
        canonical_schema = yaml.safe_load(f)

    return metadata_df, canonical_schema


def declare_metadata_from_metadata_df(metadata_df):
    tables = (
        metadata_df["Table"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    source_files = sorted(f"{t}.csv" for t in tables if t)

    return {
        "metadata": {
            "source_files": source_files
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():
    metadata_df, canonical_schema = load_mapping_context()

    parsed_schema, skipped = parse_canonical_schema(canonical_schema)

    metadata_mapping = {}

    metadata_mapping.update(
        map_metadata_column_to_canonical_field(metadata_df, parsed_schema)
    )

    if "stations" in skipped:
        stations = map_station_section(metadata_df)
        if stations:
            metadata_mapping["stations"] = stations

    if "downtime" in skipped:
        downtime = map_downtime(metadata_df)
        if downtime:
            metadata_mapping["downtime"] = downtime

    if "metrics" in skipped:
        metadata_mapping.update(declare_metrics_schema_mapping())

    metadata_mapping.update(
        declare_metadata_from_metadata_df(metadata_df)
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_MAPPING_PATH, "w") as f:
        yaml.safe_dump(metadata_mapping, f, sort_keys=False)

    print(f"\nmetadata_mapping.yml written to:\n{OUTPUT_MAPPING_PATH}")


if __name__ == "__main__":
    main()
