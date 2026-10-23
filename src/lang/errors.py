class LangError(Exception):
    """Basisklasse für Fehler der Sprache."""


class LangSyntaxError(LangError):
    pass


class LangTypeError(LangError):
    pass


class LangNameError(LangError):
    pass


class LangRuntimeError(LangError):
    pass
