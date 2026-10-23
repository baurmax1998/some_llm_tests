class LangError(Exception):
    """Basisklasse für Fehler der Sprache."""


class LangSyntaxError(LangError):
    pass


class LangIncompleteError(LangSyntaxError):
    """Das Programm ist ein gültiges Präfix, aber noch nicht vollständig."""

    def __init__(self, expected: str = "weiterer Quelltext"):
        self.expected = expected
        super().__init__(f"Unvollständiger Quelltext; erwartet {expected}")


class LangTypeError(LangError):
    pass


class LangNameError(LangError):
    pass


class LangRuntimeError(LangError):
    pass
