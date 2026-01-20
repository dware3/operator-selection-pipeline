# app/main.py
from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Any, Dict

import pandas as pd
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from pipeline.config import load_canonical_schema, validate_mapping
from pipeline.adapters.raw_builder import build_raw_run_level_table
from pipeline.mapping_wizard import new_session, next_question, apply_answer
from pipeline.artifacts import save_text_artifact, save_csv_artifact

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
app = FastAPI(title="Operator Assignment Pipeline")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DATA_DIR = os.path.join(tempfile.gettempdir(), "capstone_raw_builder")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
STATE_DIR = os.path.join(DATA_DIR, "state")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

UPLOAD_INDEX_PATH = os.path.join(STATE_DIR, "uploads.json")

# Mapping wizard session files (stateful but local)
SESSION_DIR = os.path.join(STATE_DIR, "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

CANON_SCHEMA_PATH = os.path.join(CONFIG_DIR, "canonical_schema.yml")
CANON_SCHEMA = load_canonical_schema(CANON_SCHEMA_PATH)

# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------
def _load_upload_index() -> Dict[str, Any]:
    if not os.path.exists(UPLOAD_INDEX_PATH):
        return {}
    with open(UPLOAD_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_upload_index(idx: Dict[str, Any]) -> None:
    with open(UPLOAD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


def _read_headers(csv_path: str) -> list[str]:
    df0 = pd.read_csv(csv_path, nrows=0)
    df0.columns = [str(c).strip() for c in df0.columns]
    return df0.columns.tolist()


def _read_df(file_id: str) -> pd.DataFrame:
    idx = _load_upload_index()
    rec = idx.get(file_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Unknown file_id: {file_id}")
    df = pd.read_csv(rec["path"])
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------------------------------
# Session helpers
# ------------------------------------------------------------------------------
def _session_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, f"{session_id}.json")


def _save_session(session: Dict[str, Any]) -> None:
    with open(_session_path(session["session_id"]), "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)


def _load_session(session_id: str) -> Dict[str, Any]:
    p = _session_path(session_id)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------------------
# Schema + upload API
# ------------------------------------------------------------------------------
@app.get("/schema")
def get_schema():
    """
    Canonical schema exposed so the client can drive mapping.
    """
    return {
        "schema_path": CANON_SCHEMA_PATH,
        "base_table": CANON_SCHEMA.base_table,
        "base_key": CANON_SCHEMA.base_key,
        "joiner": CANON_SCHEMA.joiner,
        "raw_csv_name": CANON_SCHEMA.raw_csv_name,
        "series_fields": CANON_SCHEMA.series_fields,
        "tables": CANON_SCHEMA.tables,
    }


@app.post("/upload")
async def upload(csv: UploadFile = File(...)):
    """
    Upload a CSV (call multiple times for multiple files).
    """
    if not csv.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv uploads are supported.")

    file_id = uuid.uuid4().hex
    dest = os.path.join(UPLOAD_DIR, f"{file_id}__{os.path.basename(csv.filename)}")

    with open(dest, "wb") as f:
        f.write(await csv.read())

    cols = _read_headers(dest)

    idx = _load_upload_index()
    idx[file_id] = {"filename": csv.filename, "path": dest, "columns": cols}
    _save_upload_index(idx)

    return {"file_id": file_id, "filename": csv.filename, "columns": cols}


@app.get("/files")
def list_files():
    idx = _load_upload_index()
    return {"files": [{"file_id": fid, **rec} for fid, rec in idx.items()]}


# ------------------------------------------------------------------------------
# Step 1: Mapping wizard -> mapping artifact
# ------------------------------------------------------------------------------
@app.post("/mapping/session/start")
def start_mapping_session():
    uploads_idx = _load_upload_index()
    if not uploads_idx:
        raise HTTPException(status_code=400, detail="No CSVs uploaded yet. POST /upload first.")

    session = new_session(CANON_SCHEMA, uploads_idx)
    _save_session(session)

    q = next_question(CANON_SCHEMA, session).__dict__
    return {"session_id": session["session_id"], "question": q}


@app.get("/mapping/session/{session_id}/next")
def mapping_next(session_id: str):
    session = _load_session(session_id)
    q = next_question(CANON_SCHEMA, session).__dict__
    return {"session_id": session_id, "question": q}


@app.post("/mapping/session/{session_id}/answer")
def mapping_answer(session_id: str, payload: Dict[str, Any]):
    """
    payload:
      {
        "table": "runs",
        "canonical_column": "run_id",
        "file_id": "...",
        "source_cols": ["Run ID"]   # list[str]
      }
    """
    session = _load_session(session_id)

    try:
        apply_answer(
            session,
            table=payload["table"],
            canonical_column=payload["canonical_column"],
            file_id=payload["file_id"],
            source_cols=payload["source_cols"],
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in payload: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _save_session(session)

    q = next_question(CANON_SCHEMA, session).__dict__
    return {"session_id": session_id, "question": q}


@app.post("/mapping/session/{session_id}/finalize")
def mapping_finalize(session_id: str):
    """
    Finalize the wizard session:
      - validate mapping against current uploads + canonical schema
      - write mapping.yaml as an artifact (mapping_config)
    """
    session = _load_session(session_id)
    if session.get("queue"):
        raise HTTPException(
            status_code=400,
            detail="Session not complete. Call /mapping/session/{id}/next and /answer until done.",
        )

    mapping_cfg = session.get("answers")
    if not isinstance(mapping_cfg, dict):
        raise HTTPException(status_code=400, detail="Session missing answers.")

    uploads_idx = _load_upload_index()
    validate_mapping(CANON_SCHEMA, mapping_cfg, uploads_idx)

    mapping_yaml = yaml.safe_dump(mapping_cfg, sort_keys=False)

    manifest = save_text_artifact(
        text=mapping_yaml,
        artifacts_dir=ARTIFACTS_DIR,
        filename="mapping.yaml",
        artifact_type="mapping_config",
        extra_manifest={"schema_path": CANON_SCHEMA_PATH},
    )
    return {"status": "ok", "mapping_artifact": manifest}


# ------------------------------------------------------------------------------
# Step 2: Automated raw build from mapping artifact -> raw CSV artifact
# ------------------------------------------------------------------------------
def _load_mapping_from_artifact(artifact_id: str) -> Dict[str, Any]:
    mapping_path = os.path.join(ARTIFACTS_DIR, artifact_id, "mapping.yaml")
    if not os.path.exists(mapping_path):
        raise HTTPException(status_code=404, detail="Mapping artifact not found or missing mapping.yaml")

    with open(mapping_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail="mapping.yaml did not parse into a dict")

    return cfg


@app.post("/raw/build")
def build_raw_from_mapping(payload: Dict[str, Any]):
    """
    payload:
      {"mapping_artifact_id": "<id>"}
    """
    mapping_artifact_id = payload.get("mapping_artifact_id")
    if not mapping_artifact_id:
        raise HTTPException(status_code=400, detail="payload must include mapping_artifact_id")

    mapping_cfg = _load_mapping_from_artifact(mapping_artifact_id)

    uploads_idx = _load_upload_index()
    validate_mapping(CANON_SCHEMA, mapping_cfg, uploads_idx)

    # Cache dfs by file_id
    dfs_by_file_id: Dict[str, pd.DataFrame] = {}
    for tmap in mapping_cfg["tables"].values():
        fid = tmap["file_id"]
        if fid not in dfs_by_file_id:
            dfs_by_file_id[fid] = _read_df(fid)

    raw_df = build_raw_run_level_table(
        schema=CANON_SCHEMA,
        mapping=mapping_cfg,
        dfs_by_file_id=dfs_by_file_id,
    )

    tmp_csv = os.path.join(tempfile.gettempdir(), f"raw_{uuid.uuid4().hex}.csv")
    raw_df.to_csv(tmp_csv, index=False)

    raw_manifest = save_csv_artifact(
        csv_path=tmp_csv,
        artifacts_dir=ARTIFACTS_DIR,
        filename=CANON_SCHEMA.raw_csv_name,
    )

    return {
        "status": "ok",
        "raw_artifact": raw_manifest,
        "rows": len(raw_df),
        "columns": list(raw_df.columns),
        "mapping_artifact_id": mapping_artifact_id,
    }


# ------------------------------------------------------------------------------
# Downloads: mapping artifact and raw artifact
# ------------------------------------------------------------------------------
@app.get("/artifacts/{artifact_id}/mapping")
def download_mapping_artifact(artifact_id: str):
    path = os.path.join(ARTIFACTS_DIR, artifact_id, "mapping.yaml")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Mapping artifact not found.")
    return FileResponse(path, media_type="text/yaml", filename="mapping.yaml")


@app.get("/artifacts/{artifact_id}/raw")
def download_raw_artifact(artifact_id: str):
    path = os.path.join(ARTIFACTS_DIR, artifact_id, CANON_SCHEMA.raw_csv_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Raw artifact not found.")
    return FileResponse(path, media_type="text/csv", filename=CANON_SCHEMA.raw_csv_name)
