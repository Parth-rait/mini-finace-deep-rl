"""Shared base types for capability inputs and outputs. Each feature keeps
its own schemas.py; the API, the agent's tool schemas and the CLI's flags
are all generated from them."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are a 422, not silently ignored


class Empty(Input):
    pass


class Health(BaseModel):
    version: str
    profile: str
    mode: str
    last_bar: str | None = Field(description="newest date in the price store, None if empty")
    staleness_bdays: int | None
    manifest_hash: str | None = Field(description="data_hash of the latest provenance manifest")
    models: list[str]
    legacy_models: list[str] = Field(description="models without a metadata card (will be refused)")
    results_rows: int
    bias_classifier: str | None
    capabilities: list[str]
