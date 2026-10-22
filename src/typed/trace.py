from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import json
import math
import numpy as np


@dataclass
class TokenProbability:
    token_id: int
    token: str
    text: str
    probability: float
    log_probability: float
    allowed: bool


@dataclass
class StepTrace:
    step: int
    selected_token_id: int
    selected_token: str
    generated_text: str
    allowed_count: int
    allowed_mass: float
    raw_top20: list[TokenProbability]
    constrained_top20: list[TokenProbability]
    sampling_top20: list[TokenProbability]
    selected_raw_probability: float
    selected_constrained_probability: float
    selected_final_probability: float
    deterministic: bool
    model_seconds: float
    mask_seconds: float


@dataclass
class GenerationTrace:
    status: str = "läuft"
    text: str = ""
    error: str | None = None
    steps: list[StepTrace] = field(default_factory=list)
    validation: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def top_tokens(probs: np.ndarray, allowed: np.ndarray, token_repr, token_text, n: int = 20) -> list[TokenProbability]:
    ids = np.flatnonzero(probs > 0)
    ids = ids[np.argsort(probs[ids])[::-1]][:n]
    return [
        TokenProbability(
            token_id=int(i),
            token=token_repr(int(i)),
            text=token_text(int(i)),
            probability=float(probs[i]),
            log_probability=math.log(float(probs[i])),
            allowed=bool(allowed[i]),
        )
        for i in ids
    ]
