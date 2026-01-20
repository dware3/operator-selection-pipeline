# app/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    columns: List[str]

class MappingTable(BaseModel):
    file_id: str
    columns: Dict[str, List[str]]  # canonical_col -> list[source cols]

class MappingConfig(BaseModel):
    tables: Dict[str, MappingTable]
