"""Numerically stable, backend-independent constrained sampling."""
from dataclasses import dataclass
import numpy as np


class ConstraintDeadEndError(RuntimeError):
    """The grammar permits no token in the model vocabulary."""


@dataclass(frozen=True)
class SamplingResult:
    token_id: int
    raw_probs: np.ndarray
    constrained_probs: np.ndarray
    sampling_probs: np.ndarray
    allowed_mass: float
    deterministic: bool


def softmax_float32(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        raise ConstraintDeadEndError("Keine endlichen Token-Kandidaten vorhanden")
    maximum = np.max(x[finite])
    exp = np.zeros_like(x, dtype=np.float32)
    exp[finite] = np.exp(x[finite] - maximum)
    return exp / np.sum(exp, dtype=np.float32)


def apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Keep the smallest descending prefix whose probability mass reaches p."""
    if not 0 < top_p <= 1:
        raise ValueError("top_p muss in (0, 1] liegen")
    result = np.asarray(logits, dtype=np.float32).copy()
    finite_ids = np.flatnonzero(np.isfinite(result))
    if finite_ids.size == 0:
        raise ConstraintDeadEndError("Top-p erhielt keine endlichen Kandidaten")
    if top_p == 1:
        return result
    probs = softmax_float32(result)
    order = finite_ids[np.argsort(probs[finite_ids])[::-1]]
    cumulative = np.cumsum(probs[order], dtype=np.float64)
    keep_count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
    result[order[keep_count:]] = -np.inf
    return result


def sample_token(
    logits: np.ndarray,
    allowed: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> SamplingResult:
    logits = np.asarray(logits, dtype=np.float32)
    allowed = np.asarray(allowed, dtype=bool)
    if logits.ndim != 1 or allowed.shape != logits.shape:
        raise ValueError("logits und allowed müssen gleich große 1-D-Arrays sein")
    if temperature < 0:
        raise ValueError("temperature darf nicht negativ sein")

    raw_probs = softmax_float32(logits)
    masked = np.where(allowed, logits, -np.inf)
    if not np.isfinite(masked).any():
        raise ConstraintDeadEndError("Die Grammatik erlaubt kein Token")
    constrained_probs = softmax_float32(masked)
    allowed_mass = float(np.sum(raw_probs[allowed], dtype=np.float64))

    if temperature == 0:
        token = int(np.argmax(masked))
        sampling_probs = np.zeros_like(raw_probs)
        sampling_probs[token] = 1.0
        deterministic = True
    else:
        sampling_logits = apply_top_p(masked / np.float32(temperature), top_p)
        sampling_probs = softmax_float32(sampling_logits)
        # Generator.choice can suffer from tiny float32 normalization drift.
        p64 = sampling_probs.astype(np.float64)
        p64 /= p64.sum()
        token = int(rng.choice(logits.size, p=p64))
        deterministic = False

    if not allowed[token]:  # defensive invariant
        raise AssertionError("Maskiertes Token wurde ausgewählt")
    return SamplingResult(token, raw_probs, constrained_probs, sampling_probs,
                          allowed_mass, deterministic)
