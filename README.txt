# Capstone-Operator-Selection-Pipeline

A schema-driven, artifact-first data engineering and ML pipeline for manufacturing analytics and operator assignment modeling.

## Overview

This system transforms heterogeneous manufacturing CSV inputs into a standardized run-level dataset using:

- Configurable mappings
- Downtime taxonomies
- Multi-operator and multi-station support
- FastAPI-based API for interaction and reproducibility

### Project Highlights

- Canonical schema and versioned artifacts
- Explicit configuration via YAML/JSON
- Aggregated, modeling-friendly run-level datasets
- API-first design using FastAPI
- Real-world, non-toy ML engineering patterns

## Repository Structure

```
Capstone-Operator-Selection-Pipeline/
├── app/                    # FastAPI application
│   ├── main.py
│   └── schemas.py
├── pipeline/
│   ├── steps/              # Modular pipeline steps
│   └── artifacts/          # Output artifacts
├── configs/                # Schema and mapping definitions
├── data/                   # Raw input data (CSV)
├── tests/                  # Unit tests
├── run_pipeline.py         # Entry point
├── requirements.txt
├── README.md
└── pytest.ini
```

## Canonical Schema

The pipeline outputs a single, run-level dataset keyed by `run_id`. Fields include:

- Runs (core data)
- Assignments (operators, stations)
- Downtime factors

Each is aggregated for compact modeling input:
- Multiple operators → `operator_1`, `operator_2`, ...
- Downtime → `downtime_factor_1`, ...

## Mapping Configuration

User-provided YAML or JSON mapping files define:

- Which CSV maps to each schema table
- How columns map to canonical names
- Which columns represent series (operators, downtime, stations)

Validated against the canonical schema for reproducibility.

## API Endpoints

Run the FastAPI service to interact with the system:

```
GET    /health
GET    /schema
POST   /upload
GET    /files
POST   /mapping
POST   /export
GET    /artifacts/{artifact_id}
```

Interactive API docs available at:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Getting Started

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn app.main:app --reload
```

## Artifacts

Each pipeline run produces one or more CSV artifacts under pipeline/artifacts/, including:

-canonical_flat.csv — flattened canonical run-level data
-canonical_wide_validated.csv — validated wide-format dataset
-operator_product_downtime.csv — operator-level downtime aggregation
-operator_downtime_variability.csv — operator variability metrics
-operator_factor_tests.csv — statistical test results
-station_operator_assignments.csv — station-to-operator mappings
-product.product_code_operator_assignments.csv — product-level operator assignments
-metadata_mapping.yml — configuration used for the transformation

These artifacts are designed for downstream ML modeling and analysis. Each file represents a different view or aggregation of the canonical manufacturing data.

## Design Principles

- **Schema-first**: Canonical format drives logic, not ad-hoc datasets
- **Configurable**: All mappings are user-defined and versioned
- **Reproducible**: Every transformation is auditable and deterministic
- **Separated concerns**: Data prep ≠ feature engineering ≠ modeling

## License

This project is licensed under the MIT License.
