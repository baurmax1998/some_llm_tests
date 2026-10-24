"""MLX model/tokenizer lifecycle and incremental KV-cache inference."""
from __future__ import annotations
from typing import Any
import numpy as np

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_SYSTEM_PROMPT = "Antworte ausschließlich mit dem verlangten JSON-Dokument."


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
        self._processed_ids: list[int] = []
        self._started = False
        self.last_cache_reused = False
        self.last_prompt_token_count = 0
        self._prepared_chat_text: str | None = None
        self.last_generation_prefix = ""
        self.last_cache_miss_reason = "not_requested"

    def _message_text(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool = True
    ) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            # Tokenizers without the Qwen option have no thinking mode to disable.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )

    def _encode_text(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def prepare_messages(
        self,
        messages: list[dict[str, str]],
        *,
        reuse_cache: bool = False,
        previous_response: str | None = None,
        previous_response_token_ids: list[int] | None = None,
    ) -> bool:
        """Bereitet einen Chat vor und verwendet einen passenden KV-Präfix erneut.

        ``messages`` muss bei Wiederverwendung die bisherige Unterhaltung als
        unveränderten Präfix enthalten. Antworttext und tatsächlich generierte
        Token-IDs schließen dabei die noch nicht verarbeitete Antwort sauber ab.
        """
        full_text = self._message_text(messages)
        # Einige Templates setzen vor der sichtbaren Antwort einen festen
        # Thinking-Präfix ein. Als historische Assistant-Nachricht muss dieser
        # Präfix erhalten bleiben, damit Text und gecachte Token identisch sind.
        probe = "__PI_ASSISTANT_CONTENT__"
        assistant_probe = self._message_text(
            [*messages, {"role": "assistant", "content": probe}],
            add_generation_prompt=False,
        )
        probe_position = assistant_probe.rfind(probe)
        assistant_header = assistant_probe[:probe_position] if probe_position >= 0 else full_text
        self.last_generation_prefix = (
            full_text[len(assistant_header):]
            if full_text.startswith(assistant_header)
            else ""
        )
        response_ids = previous_response_token_ids or []
        expected_text = (
            (self._prepared_chat_text or "") + (previous_response or "")
        )
        suffix: str | None = None
        if full_text.startswith(expected_text):
            suffix = full_text[len(expected_text):]
        elif previous_response is not None and len(messages) >= 2:
            # Qwen entfernt bei historischen Assistant-Nachrichten den leeren
            # Thinking-Präfix. Der Cache enthält ihn trotzdem. Deshalb wird nur
            # der Template-Suffix nach der sichtbaren Antwort angehängt.
            response_position = full_text.rfind(previous_response)
            if response_position >= 0:
                suffix = full_text[response_position + len(previous_response):]
        checks = {
            "requested": reuse_cache,
            "processed": bool(self._processed_ids),
            "response_tokens": bool(response_ids),
            "prepared_chat": self._prepared_chat_text is not None,
            "chat_suffix": suffix is not None,
        }
        reusable = all(checks.values())
        self.last_cache_miss_reason = "reused" if reusable else ",".join(
            name for name, passed in checks.items() if not passed
        )
        if reusable:
            # Der letzte ausgewählte Antwort-Token wurde noch nicht durch das
            # Modell geschickt. Danach folgt nur der neue Chat-Template-Suffix.
            self._prompt_ids = [
                response_ids[-1],
                *self._encode_text(suffix or "", add_special_tokens=False),
            ]
        else:
            self.cache = self._make_prompt_cache(self.model)
            self._processed_ids = []
            add_special = self.tokenizer.bos_token is None or not full_text.startswith(self.tokenizer.bos_token)
            self._prompt_ids = self._encode_text(full_text, add_special_tokens=add_special)
        self._prepared_chat_text = full_text
        self._started = False
        self.last_cache_reused = reusable
        self.last_prompt_token_count = len(self._prompt_ids)
        return reusable

    def prepare_prompt(self, user_prompt: str, system_prompt: str | None = None) -> None:
        # Normale Einzelanfragen sind unabhängig und übernehmen keinen KV-Zustand.
        self.prepare_messages([
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

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
        self._processed_ids.extend(int(token_id) for token_id in ids)
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
