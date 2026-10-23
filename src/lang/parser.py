"""Parser für die in ``konzept.md`` beschriebene Ausdruckssprache."""
from __future__ import annotations
from dataclasses import replace

from .ast import (
    Annotation, Binary, Binding, Call, Expr, Field, IfExpr, Lambda, ListExpr,
    Literal, MethodCall, Name, Program, RecordExpr, SourceSpan, TypeRef, Unary,
)
from .errors import LangIncompleteError, LangSyntaxError
from .lexer import Token, lex


_PRECEDENCE = {"or": 1, "and": 2, "==": 3, "!=": 3, "<": 3, "<=": 3,
               ">": 3, ">=": 3, "+": 4, "-": 4, "*": 5}


class Parser:
    def __init__(self, source: str, allow_incomplete: bool = False):
        self.allow_incomplete = allow_incomplete
        self.tokens = lex(source, allow_incomplete=allow_incomplete)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def match(self, kind: str, value: str | None = None) -> bool:
        if self.current.kind == kind and (value is None or self.current.value == value):
            self.advance()
            return True
        return False

    def expect(self, kind: str, value: str | None = None) -> Token:
        if not self.match(kind, value):
            wanted = value if value is not None else kind
            if self.allow_incomplete and self.current.kind == "EOF":
                raise LangIncompleteError(repr(wanted))
            if (
                self.allow_incomplete
                and value is not None
                and self.current.kind == kind
                and value.startswith(self.current.value)
                and self.index + 1 < len(self.tokens)
                and self.tokens[self.index + 1].kind == "EOF"
            ):
                raise LangIncompleteError(repr(value))
            raise LangSyntaxError(
                f"Erwartet {wanted!r}, gefunden {self.current.value or self.current.kind!r}",
                self.current.offset,
                self.current.offset + max(1, len(self.current.value)),
            )
        return self.tokens[self.index - 1]

    def parse_program(self) -> Program:
        bindings: list[Binding] = []
        while self.current.kind != "EOF":
            binding_start = self.current.offset
            keyword = self.expect("NAME").value
            if keyword not in {"goal", "let"}:
                if (
                    self.allow_incomplete
                    and any(candidate.startswith(keyword) for candidate in ("goal", "let"))
                    and self.current.kind == "EOF"
                ):
                    raise LangIncompleteError("'goal' oder 'let'")
                token = self.tokens[self.index - 1]
                raise LangSyntaxError(
                    f"Erwartet 'goal' oder 'let', gefunden {keyword!r}",
                    token.offset, token.offset + len(token.value),
                )
            name = self.expect("NAME").value
            annotation = None
            if self.match(":"):
                annotation = self.parse_type()
            self.expect("=")
            value = self.parse_expression()
            bindings.append(Binding(
                name, value, annotation, keyword == "goal",
                span=SourceSpan(binding_start, self.current.offset),
            ))
        goals = sum(binding.is_goal for binding in bindings)
        if goals != 1:
            if self.allow_incomplete and goals == 0:
                raise LangIncompleteError("eine goal-Definition")
            raise LangSyntaxError("Ein Programm muss genau ein goal enthalten")
        names = [binding.name for binding in bindings]
        duplicate = next((name for name in names if names.count(name) > 1), None)
        if duplicate:
            raise LangSyntaxError(f"Name {duplicate!r} wird mehrfach definiert")
        return Program(tuple(bindings))

    def parse_type(self) -> TypeRef:
        name = self.expect("NAME").value
        argument = None
        if self.match("OP", "<"):
            argument = self.parse_type()
            self.expect("OP", ">")
        if name == "List" and argument is None:
            if self.allow_incomplete and self.current.kind == "EOF":
                raise LangIncompleteError("'<' und einen Elementtyp")
            raise LangSyntaxError("List benötigt einen Elementtyp")
        if name != "List" and argument is not None:
            raise LangSyntaxError(f"Nur List darf einen Typparameter besitzen, nicht {name}")
        return TypeRef(name, argument)

    def parse_expression(self, minimum: int = 0) -> Expr:
        start = self.current.offset
        left = self.parse_prefix()
        left = self.parse_postfix(left)
        while True:
            if self.allow_incomplete and self.current.kind == "PARTIAL_OP":
                raise LangIncompleteError("'!='")
            if self.allow_incomplete and self.current.kind == "=":
                raise LangIncompleteError("'=='")
            operator = self.current.value
            precedence = _PRECEDENCE.get(operator, -1)
            if precedence < minimum:
                break
            self.advance()
            right = self.parse_expression(precedence + 1)
            left = Binary(operator, left, right)
        return replace(left, span=SourceSpan(start, self.current.offset))

    def parse_prefix(self) -> Expr:
        token = self.current
        if token.kind == "INT":
            self.advance()
            return Literal(int(token.value))
        if token.kind == "STRING":
            self.advance()
            return Literal(token.value)
        if token.kind == "NAME" and token.value in {"true", "false"}:
            self.advance()
            return Literal(token.value == "true")
        if token.kind == "NAME" and token.value == "if":
            self.advance()
            condition = self.parse_expression()
            self.expect("NAME", "then")
            then_branch = self.parse_expression()
            self.expect("NAME", "else")
            return IfExpr(condition, then_branch, self.parse_expression())
        if token.kind == "NAME" and token.value == "not":
            self.advance()
            return Unary("not", self.parse_expression(6))
        if token.kind == "OP" and token.value == "-":
            self.advance()
            return Unary("-", self.parse_expression(6))
        if token.kind == "NAME":
            self.advance()
            if self.match("ARROW"):
                return Lambda((token.value,), self.parse_expression())
            if (
                self.allow_incomplete
                and self.current.kind == "="
                and self.index + 1 < len(self.tokens)
                and self.tokens[self.index + 1].kind == "EOF"
            ):
                raise LangIncompleteError("'=>'")
            origin: list[str] = []
            while self.match("ORIGIN"):
                origin.append(self.expect("NAME").value)
            return Name(token.value, tuple(origin))
        if self.match("["):
            items: list[Expr] = []
            if not self.match("]"):
                while True:
                    items.append(self.parse_expression())
                    if self.match("]"):
                        break
                    self.expect(",")
            return ListExpr(tuple(items))
        if self.match("{"):
            fields: list[tuple[str, Expr]] = []
            if not self.match("}"):
                while True:
                    name = self.expect("NAME").value
                    self.expect(":")
                    fields.append((name, self.parse_expression()))
                    if self.match("}"):
                        break
                    self.expect(",")
            return RecordExpr(tuple(fields))
        if self.match("("):
            # Parameterlisten von Lambdas haben nur Namen und enden direkt vor =>.
            saved = self.index
            parameters: list[str] = []
            if self.current.kind == "NAME":
                parameters.append(self.advance().value)
                while self.match(","):
                    parameters.append(self.expect("NAME").value)
                if self.match(")") and self.match("ARROW"):
                    return Lambda(tuple(parameters), self.parse_expression())
            self.index = saved
            value = self.parse_expression()
            if self.match(":"):
                value = Annotation(value, self.parse_type())
            self.expect(")")
            return value
        if self.allow_incomplete and token.kind in {"EOF", "PARTIAL_STRING"}:
            raise LangIncompleteError("einen Ausdruck")
        raise LangSyntaxError(
            f"Ausdruck erwartet, gefunden {token.value or token.kind!r}",
            token.offset, token.offset + max(1, len(token.value)),
        )

    def parse_postfix(self, value: Expr) -> Expr:
        while True:
            if self.match("("):
                arguments = self.parse_arguments_after_open()
                value = Call(value, arguments)
            elif self.match("."):
                name = self.expect("NAME").value
                if self.match("("):
                    value = MethodCall(value, name, self.parse_arguments_after_open())
                else:
                    value = Field(value, name)
            elif self.match(":"):
                # Typannotationen sind Postfix-Ausdrücke. Dadurch funktionieren
                # Ketten wie `customers: List<Customer>.map(...)` ohne Klammern.
                value = Annotation(value, self.parse_type())
            else:
                return value

    def parse_arguments_after_open(self) -> tuple[Expr, ...]:
        arguments: list[Expr] = []
        if not self.match(")"):
            while True:
                arguments.append(self.parse_expression())
                if self.match(")"):
                    break
                self.expect(",")
        return tuple(arguments)


def parse(source: str) -> Program:
    return Parser(source).parse_program()
