from __future__ import annotations
import time
from typing import Callable, Any
import numpy as np
from ..mask.sampling import sample_token
from ..typed.trace import GenerationTrace, StepTrace, top_tokens
from ..typed.validation import validate_output


def generate(
    backend: Any,
    constraint: Any,
    schema: dict,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    max_new_tokens: int = 256,
    on_step: Callable[[GenerationTrace], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> GenerationTrace:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens muss positiv sein")
    trace = GenerationTrace()
    rng = np.random.default_rng(seed)
    generated_ids: list[int] = []
    previous = None
    backend.prepare_prompt(prompt)

    try:
        for step in range(max_new_tokens):
            if should_cancel and should_cancel():
                trace.status = "abgebrochen"
                return trace
            started = time.perf_counter()
            logits = backend.next_logits(previous)
            model_seconds = time.perf_counter() - started
            started = time.perf_counter()
            allowed = constraint.allowed_token_mask()
            mask_seconds = time.perf_counter() - started
            if allowed.shape != logits.shape:
                raise RuntimeError(f"Vokabulargrößen stimmen nicht überein: {allowed.size} != {logits.size}")
            result = sample_token(logits, allowed, temperature, top_p, rng)
            token = result.token_id
            constraint.accept_token(token)
            generated_ids.append(token)
            text = backend.decode(generated_ids)
            trace.text = text
            trace.steps.append(StepTrace(
                step=step, selected_token_id=token, selected_token=backend.token_repr(token),
                generated_text=text, allowed_count=int(allowed.sum()), allowed_mass=result.allowed_mass,
                raw_top20=top_tokens(result.raw_probs, allowed, backend.token_repr, backend.token_text),
                constrained_top20=top_tokens(result.constrained_probs, allowed, backend.token_repr, backend.token_text),
                sampling_top20=top_tokens(result.sampling_probs, allowed, backend.token_repr, backend.token_text),
                selected_raw_probability=float(result.raw_probs[token]),
                selected_constrained_probability=float(result.constrained_probs[token]),
                selected_final_probability=float(result.sampling_probs[token]),
                deterministic=result.deterministic, model_seconds=model_seconds, mask_seconds=mask_seconds,
            ))
            if on_step:
                on_step(trace)
            if constraint.is_complete():
                value = validate_output(text, schema)
                trace.validation = {"json": True, "jsonschema": True, "value": value}
                trace.status = "abgeschlossen"
                return trace
            previous = token
        trace.status = "unvollständig"
        trace.error = "Tokenlimit erreicht, bevor die Grammatik einen akzeptierenden Zustand erreichte"
    except Exception as exc:
        trace.status = "fehlgeschlagen"
        trace.error = f"{type(exc).__name__}: {exc}"
    return trace
