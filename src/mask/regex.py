"""Tokenweiser Constraint für kleine reguläre Textformate."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import SchemaConstraint


class OutlinesRegexConstraint(SchemaConstraint):
    """Beschränkt eine Generierung auf einen regulären Ausdruck."""

    def __init__(self, regex: str, tokenizer: Any, vocab_size: int | None = None) -> None:
        if not regex:
            raise ValueError("regex darf nicht leer sein")
        from outlines_core import Guide, Index, Vocabulary

        vocab = tokenizer.get_vocab()
        eos_id = tokenizer.eos_token_id
        eos_token = tokenizer.eos_token
        if eos_id is None or eos_token is None:
            raise ValueError("Tokenizer benötigt eos_token und eos_token_id")
        formatted: dict[str, list[int]] = {}
        for token, token_id in vocab.items():
            text = tokenizer.convert_tokens_to_string([token])
            formatted.setdefault(text, []).append(int(token_id))
        formatted.pop(tokenizer.convert_tokens_to_string([eos_token]), None)
        vocabulary = Vocabulary(int(eos_id), formatted)
        self._guide = Guide(Index(regex, vocabulary))
        tokenizer_size = max(int(token_id) for token_id in vocab.values()) + 1
        self._vocab_size = vocab_size or tokenizer_size
        if self._vocab_size < tokenizer_size:
            raise ValueError("Modellvokabular ist kleiner als das Tokenizervokabular")

    def allowed_token_mask(self) -> np.ndarray:
        mask = np.zeros(self._vocab_size, dtype=bool)
        mask[np.asarray(self._guide.get_tokens(), dtype=np.int64)] = True
        return mask

    def accept_token(self, token_id: int) -> None:
        if not self._guide.accepts_tokens([int(token_id)]):
            raise ValueError(f"Token {token_id} ist im aktuellen Regexzustand unzulässig")
        self._guide.advance(int(token_id), return_tokens=False)

    def is_complete(self) -> bool:
        return bool(self._guide.is_finished())
