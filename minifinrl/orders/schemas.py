"""Inputs and outputs of the orders capabilities (declared, never executed)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from minifinrl.platform.schemas import Input


class OrderIn(Input):
    ticker: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)


class OrderOut(BaseModel):
    placed: bool
