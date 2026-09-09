"""Request models for the small HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionRequest(BaseModel):
    session_id: str | None = None
    width: int = Field(default=128, ge=8, le=1024)
    height: int = Field(default=128, ge=8, le=1024)
    density: float = Field(default=0.20, ge=0.0, le=1.0)
    # ``null`` requests a freshly generated seed; the response always returns
    # the concrete integer used to initialize the grid.
    seed: int | None = 42
    rule: str = "B3/S23"


class Session3DRequest(BaseModel):
    session_id: str | None = None
    depth: int = Field(default=32, ge=8, le=128)
    height: int = Field(default=32, ge=8, le=128)
    width: int = Field(default=32, ge=8, le=128)
    density: float = Field(default=0.04, ge=0.0, le=1.0)
    seed: int | None = 42
    rule: str = "B6/S5,6,7"


class ActionRequest(BaseModel):
    session_id: str = "default"


class StepRequest(ActionRequest):
    steps: int = Field(default=1, ge=1, le=100_000)
    collect_metrics: bool = False


class StabilityRequest(ActionRequest):
    max_steps: int = Field(default=100, ge=0, le=100_000)


class RandomizeRequest(ActionRequest):
    seed: int | None = 42
    density: float = Field(default=0.20, ge=0.0, le=1.0)


class RuleRequest(ActionRequest):
    rule: str


class RandomRuleRequest(ActionRequest):
    seed: int | None = None


class StateRequest(ActionRequest):
    grid: list[list[int]]
    rule: str | None = None
    seed: int | None = None
    density: float | None = Field(default=None, ge=0.0, le=1.0)
    generation: int = Field(default=0, ge=0)


class Experiment3DRequest(BaseModel):
    rules: int = Field(default=5, ge=1, le=100)
    initial_conditions: int = Field(default=4, ge=1, le=100)
    size: int = Field(default=24, ge=8, le=96)
    steps: int = Field(default=50, ge=0, le=2_000)
    density: float = Field(default=0.10, ge=0.0, le=1.0)
    seed: int = 42


class Experiment2DRequest(BaseModel):
    rules: int = Field(default=5, ge=1, le=100)
    initial_conditions: int = Field(default=4, ge=1, le=100)
    size: int = Field(default=24, ge=8, le=256)
    steps: int = Field(default=50, ge=0, le=2_000)
    density: float = Field(default=0.10, ge=0.0, le=1.0)
    seed: int = 42
