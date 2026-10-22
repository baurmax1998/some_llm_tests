"""MLX model/tokenizer lifecycle and incremental KV-cache inference."""
from __future__ import annotations
from typing import Any
import numpy as np

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"


class MLXModelBackend:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as exc:
            raise RuntimeError("MLX-Abhängigkeiten fehlen; siehe requirements.txt") from exc
        self.mx = mx
        self.model, self.tokenizer = load(model_id)
        self._make_prompt_cache = make_prompt_cache
        self.cache = make_prompt_cache(self.model)
        self._prompt_ids: list[int] | None = None
        self._started = False

    def prepare_prompt(self, user_prompt: str) -> None:
        # A backend instance can run several independent requests. Never carry
        # generated KV state from one schema/example into the next.
        self.cache = self._make_prompt_cache(self.model)
        messages = [
            {"role": "system", "content": "Antworte ausschließlich mit dem verlangten JSON-Dokument."},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except (TypeError, ValueError):
            # Tokenizers without the Qwen option have no thinking mode to disable.
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        add_special = self.tokenizer.bos_token is None or not text.startswith(self.tokenizer.bos_token)
        self._prompt_ids = list(self.tokenizer.encode(text, add_special_tokens=add_special))
        self._started = False

    def next_logits(self, previous_token: int | None = None) -> np.ndarray:
        if self._prompt_ids is None:
            raise RuntimeError("prepare_prompt muss zuerst aufgerufen werden")
        if not self._started:
            ids = self._prompt_ids
            self._started = True
        else:
            if previous_token is None:
                raise ValueError("Nach dem Prefill wird previous_token benötigt")
            ids = [previous_token]
        x = self.mx.array(ids)[None]
        logits = self.model(x, cache=self.cache)[:, -1, :]
        # NumPy cannot consume MLX bfloat16 through PEP 3118. Cast on the
        # MLX side before crossing the array boundary.
        logits = logits.astype(self.mx.float32)
        self.mx.eval(logits)
        return np.array(logits[0], dtype=np.float32, copy=True)

    @property
    def vocab_size(self) -> int:
        # Quantized models may pad the output head beyond the tokenizer IDs.
        args = getattr(self.model, "args", None)
        return int(getattr(args, "vocab_size", max(self.tokenizer.get_vocab().values()) + 1))

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def token_text(self, token_id: int) -> str:
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        return self.tokenizer.convert_tokens_to_string([token])

    def token_repr(self, token_id: int) -> str:
        return repr(self.tokenizer.convert_ids_to_tokens(int(token_id)))
