import abc
import copy
import string
import sys
from collections.abc import Sequence
from typing import Iterable, Iterator

from genrex.logging import logger
from genrex.misc import ready_to_print, string2ngrams
from genrex.types import (
    ClassType,
    class_types_array,
    define_type,
    hex_lowercase,
    hex_uppercase,
)

find_keywords = [
    "administrator",
    "application",
    "assert",
    "breadcrumb",
    "cho",
    "chrome",
    "days",
    "desktop",
    "dialog",
    "documents",
    "edge",
    "frame",
    "google",
    "java",
    "manager",
    "mutex",
    "processor",
    "program",
    "script",
    "server",
    "spoolsv",
    "svchost",
    "synaptics",
    "task",
    "toolbar",
    "txt",
    "ver",
    "window",
    "windows",
    "genrexdot",
    "genrexbracket",
    "genrexguid",
    "genrexhex",
    "genrexhex8",
    "genrexexem",
]


class Expression(abc.ABC):
    suffix: bool = False

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator["Expression"]:
        # iterator over possible alternatives
        pass

    @abc.abstractmethod
    def remove(self, end: str):
        pass

    @abc.abstractmethod
    def heuristic(self) -> "Expression":
        pass

    @abc.abstractmethod
    def return_class_repr(self) -> ClassType | None:
        pass

    @abc.abstractmethod
    def min_max(self) -> tuple[int, int]:
        pass

    @abc.abstractmethod
    def alt_len(self) -> int:
        pass

    @abc.abstractmethod
    def endswith(self, end: str) -> bool:
        pass

    @abc.abstractmethod
    def get_ngrams(self, ngram_len: int) -> list[str]:
        pass

    def split(self) -> list["Expression"]:
        return [self]

    @classmethod
    def join(cls, values: list["Expression"]):
        if not values:
            return Literal("")
        result = values.pop(0)
        while values:
            result = Concatenation(result, values.pop(0))
        return result

    @abc.abstractmethod
    def contains_meta(self, meta: "Meta") -> bool:
        return False

    @abc.abstractmethod
    def reduce(self) -> "Expression":
        pass


class Literal(Expression):
    def __init__(self, literal: str):
        self.literal = literal
        self.i = 0

    def __eq__(self, other) -> bool:
        return isinstance(other, Literal) and self.literal == other.literal

    def __hash__(self):
        return hash(self.literal)

    def __repr__(self) -> str:
        return self.literal

    def __str__(self) -> str:
        res = self.literal
        res = res.replace("\\", r"\\")
        for char in [
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "^",
            "$",
            "|",
            "*",
            "+",
            ".",
            "?",
            "/",
        ]:
            if char in res:
                res = res.replace(char, "\\" + char)
        if r"\\\[\^\\\]\+\\" in res:
            res = res.replace(r"\\\[\^\\\]\+\\", r"\\[^\\]+\\")

        res = ready_to_print(res)

        return res

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < 1:
            self.i += 1
            return self
        else:
            self.i = 0
            raise StopIteration

    def __len__(self) -> int:
        return len(self.literal)

    def alt_len(self) -> int:
        return 1

    def min_max(self) -> tuple[int, int]:
        return len(self), len(self)

    def endswith(self, end: str) -> bool:
        return self.literal.endswith(end)

    def remove(self, end: str):
        if self.literal.endswith(end):
            self.literal = self.literal[: -len(end)]
        return self

    def add(self, end: str):
        self.literal = self.literal + end

    def return_class_repr(self) -> ClassType | None:
        return define_type(self.literal)

    def heuristic(self) -> Expression:
        return self

    def get_ngrams(self, ngram_len: int) -> list[str]:
        return string2ngrams(self.literal, ngram_len)

    def contains_meta(self, meta: "Meta") -> bool:
        return False

    def reduce(self) -> Expression:
        return self


class Class(Expression):
    # Don't create  instance of Class that contains duplicitous Literals/Meta
    def __init__(self, class_set: Iterable[Expression]):
        # the class is set (unique characters only)
        self.class_set = list(set(class_set))
        self.i = 0

    def __eq__(self, other) -> bool:
        return isinstance(other, Class) and sorted(self.class_set, key=repr) == sorted(
            other.class_set, key=repr
        )

    def __hash__(self):
        return hash(tuple(sorted(self.class_set, key=repr)))

    def __repr__(self) -> str:
        class_set_sorted = sorted(self.class_set, key=repr)
        res = ""
        for expr in class_set_sorted:
            res += repr(expr)
        return res

    def __str__(self) -> str:
        class_set_sorted = sorted(self.class_set, key=repr)
        res = ""
        for expr in class_set_sorted:
            expr_str = str(expr)
            # lit: ["(", ")", "[", "]", "{", "}", "^", "$", "|", "*", "+", ".", "?"]:
            if expr_str in [
                r"\(",
                r"\)",
                r"\{",
                r"\}",
                r"\$",
                r"\|",
                r"\*",
                r"\+",
                r"\.",
                r"\?",
            ]:
                expr_str = expr_str[-1]
            # escaped in class: - (^, [, ], \, / are done by Literal)
            if expr_str in ["-"]:
                expr_str = "\\" + expr_str
            if expr_str == r"\\s":
                expr_str = r"\s"
            res += str(expr_str)

        if len(res) == 1 or res == r"\s":
            return res
        else:
            return "[" + res + "]"

    def __iter__(self) -> Iterator[Expression]:
        return self

    def __next__(self):
        if self.i < len(self):
            self.i += 1
            return self.class_set[self.i - 1]
        else:
            self.i = 0
            raise StopIteration

    def __len__(self) -> int:
        return len(self.class_set)

    def alt_len(self) -> int:
        return 1

    def min_max(self) -> tuple[int, int]:
        return 1, 1

    def endswith(self, end: str) -> bool:
        return False

    def myord(self, char) -> int:
        repr_char = repr(char)
        if len(repr_char) == 0:
            return 0
        elif len(repr_char) == 1:
            return ord(repr_char)
        else:
            return ord(repr_char[-1])

    def return_class_repr(self) -> ClassType | None:
        return define_type(repr(self))

    def heuristic(self) -> Expression:
        if len(self) > 2:
            class_set = self.simplify()
            if class_set:
                self.class_set = class_set

        class_set_sorted = sorted(self.class_set, key=str)
        res = []
        res_len = len(class_set_sorted) - 1
        i = 0
        while i < res_len:
            if self.myord(class_set_sorted[i]) + 1 == self.myord(
                class_set_sorted[i + 1]
            ):
                j = i + 1
                while j < res_len:
                    if self.myord(class_set_sorted[j]) + 1 == self.myord(
                        class_set_sorted[j + 1]
                    ):
                        j += 1
                    else:
                        break
                if j == i + 1:
                    res.append(class_set_sorted[i])
                else:
                    res.append(
                        Literal(
                            repr(class_set_sorted[i]) + "-" + repr(class_set_sorted[j])
                        )
                    )
                    i = j
            else:
                res.append(class_set_sorted[i])
            i += 1
        if i == res_len:
            res.append(class_set_sorted[i])
        self.class_set = res
        return self

    def simplify(self) -> list[Expression]:
        class_set: set[Expression] = set()
        res_type = self.return_class_repr()
        if res_type is not None and res_type != ClassType.GENERAL:
            for class_type in ClassType:
                if class_type & res_type:
                    class_set.update(
                        Literal(char) for char in class_types_array[class_type]
                    )
        return list(class_set)

    def remove(self, end: str):
        logger.warning(f"Remove class used: {end} {self.class_set}")
        if end in self.class_set:
            self.class_set.remove(end)  # type: ignore #
        return self

    def get_ngrams(self, ngram_len: int) -> list[str]:
        return []

    def contains_meta(self, meta: "Meta") -> bool:
        return meta in self.class_set

    def reduce(self) -> Expression:
        return self


class Concatenation(Expression):
    # This class should not be used to connect two Literals,
    # but instead they should be connected into one Literal
    def __init__(self, expr_a: Expression, expr_b: Expression):
        self.expr_a: Expression = expr_a
        self.expr_b: Expression = expr_b
        self.i: int = 0

    def __eq__(self, other) -> bool:
        return isinstance(other, Concatenation) and str(self) == str(other)

    def __hash__(self):
        return hash((self.expr_a, self.expr_b))

    def __repr__(self) -> str:
        return repr(self.expr_a) + repr(self.expr_b)

    def __str__(self) -> str:
        return str(self.expr_a) + str(self.expr_b)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < 1:
            self.i += 1
            return self
        else:
            self.i = 0
            raise StopIteration

    def __len__(self) -> int:
        return len(self.expr_a) + len(self.expr_b)

    def alt_len(self) -> int:
        return self.expr_a.alt_len() + self.expr_b.alt_len() - 1

    def min_max(self) -> tuple[int, int]:
        min_a, max_a = self.expr_a.min_max()
        min_b, max_b = self.expr_b.min_max()
        return min_a + min_b, max_a + max_b

    def endswith(self, end: str) -> bool:
        return self.expr_b.endswith(end)

    def remove(self, end: str):
        self.expr_b = self.expr_b.remove(end)
        if len(self.expr_b) == 0:
            return self.expr_a
        return self

    def return_class_repr(self) -> ClassType | None:
        if self.expr_a.min_max() == (0, 0):
            return self.expr_b.return_class_repr()

        if self.expr_b.min_max() == (0, 0):
            return self.expr_a.return_class_repr()

        expr_a = self.expr_a.return_class_repr()
        expr_b = self.expr_b.return_class_repr()

        if expr_a is None or expr_b is None:
            return None

        return expr_a | expr_b

    def heuristic(self) -> Expression:
        if isinstance(self.expr_a, Literal):
            if isinstance(self.expr_b, Alternation):
                self.test_keywords_prefix()
                if len(self.expr_a) == 0:
                    return self.expr_b.heuristic()
            elif isinstance(self.expr_b, Concatenation) and isinstance(
                self.expr_b.expr_a, Alternation
            ):
                rest = self.expr_b.expr_b
                self.expr_b = self.expr_b.expr_a
                self.test_keywords_prefix()
                self.expr_b = Concatenation(self.expr_b, rest)
                if len(self.expr_a) == 0:
                    return self.expr_b.heuristic()
        elif isinstance(self.expr_b, Literal) and isinstance(self.expr_a, Alternation):
            self.test_keywords_suffix()
            if len(self.expr_b) == 0:
                return self.expr_a.heuristic()

        self.expr_a = self.expr_a.heuristic()
        self.expr_b = self.expr_b.heuristic()
        return self

    def test_keywords_prefix(self):
        len_of_a = len(repr(self.expr_a))
        if len_of_a == 0:
            return
        test_it = min(len_of_a, 3)
        for i in range(1, test_it + 1):
            rep_a = repr(self.expr_a)[len_of_a - i :]
            for alternation in self.expr_b.expr_alternations:  # type: ignore
                if any(
                    (rep_a + repr(alternation)).lower().startswith(y)
                    for y in find_keywords
                ):
                    if len_of_a > i:
                        self.expr_a = Literal(repr(self.expr_a)[: len_of_a - i])
                        self.expr_b = Alternation(
                            Literal(rep_a + repr(alternation))
                            for alternation in self.expr_b.expr_alternations
                        )
                    else:
                        self.expr_a = Literal("")
                        self.expr_b = Alternation(
                            Literal(rep_a + repr(alternation))
                            for alternation in self.expr_b.expr_alternations
                        )
                    return

    def test_keywords_suffix(self):
        len_of_b = len(repr(self.expr_b))
        if len_of_b == 0:
            return
        rep_b = repr(self.expr_b)[0]
        for alternation in self.expr_a.expr_alternations:  # type: ignore
            if any(
                (repr(alternation) + rep_b).lower().endswith(y) for y in find_keywords
            ):
                if len_of_b > 1:
                    self.expr_a = Alternation(
                        [
                            Literal(repr(alternation) + rep_b)
                            for alternation in self.expr_a.expr_alternations
                        ]
                    )
                    self.expr_b = Literal(repr(self.expr_b)[1:])
                else:
                    self.expr_a = Alternation(
                        [
                            Literal(repr(alternation) + rep_b)
                            for alternation in self.expr_a.expr_alternations
                        ]
                    )
                    self.expr_b = Literal("")
                return

    def get_ngrams(self, ngram_len: int) -> list[str]:
        return self.expr_a.get_ngrams(ngram_len) + self.expr_b.get_ngrams(ngram_len)

    def get_left(self) -> Expression:
        if isinstance(self.expr_a, Concatenation):
            return self.expr_a.get_left()
        return self.expr_a

    def get_right(self) -> Expression:
        if isinstance(self.expr_b, Concatenation):
            return self.expr_b.get_right()
        return self.expr_b

    def split(self) -> list["Expression"]:
        return [*self.expr_a.split(), *self.expr_b.split()]

    def contains_meta(self, meta: "Meta") -> bool:
        return self.expr_a.contains_meta(meta) or self.expr_b.contains_meta(meta)

    def reduce(self) -> Expression:
        self.expr_a = self.expr_a.reduce()
        self.expr_b = self.expr_b.reduce()
        return self


class Alternation(Expression):
    # Don't create instance of Alternation that contains duplicitous Literals/Meta

    def __init__(self, alternations: Iterable[Expression]):
        self.expr_alternations: list[Expression] = sorted(alternations, key=repr)
        self.i = 0

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Alternation)
            and self.expr_alternations == other.expr_alternations
        )

    def __hash__(self):
        return hash(tuple(sorted(self.expr_alternations, key=repr)))

    def get_list(self):
        return self.expr_alternations

    def __repr__(self) -> str:
        if self.expr_alternations is None:
            return ""
        alternations_sorted = sorted(self.expr_alternations, key=repr)
        res = []
        for alternation in alternations_sorted:
            if len(alternation) == 0:
                continue
            res.append(repr(alternation))
        return "|".join(res)

    def __str__(self) -> str:
        if self.expr_alternations is None:
            return ""
        alternations_sorted = sorted(self.expr_alternations, key=repr)
        res = []
        maybe = ""
        for alternation in alternations_sorted:
            if len(alternation) == 0:
                maybe = "?"
                continue
            if isinstance(alternation, Alternation):
                alt = str(alternation)[1:-1]
            else:
                alt = str(alternation)
            res.append(alt)
        if len(self.expr_alternations) == 1:
            return res[0]
        return "(" + "|".join(res) + ")" + maybe

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < len(self):
            self.i += 1
            return self.expr_alternations[self.i - 1]
        else:
            self.i = 0
            raise StopIteration

    def __len__(self) -> int:
        if self.expr_alternations is None:
            return 0
        return len(self.expr_alternations)

    def alt_len(self) -> int:
        if self.expr_alternations is None:
            return 0
        else:
            res = 0
            for alternation in self.expr_alternations:
                res += alternation.alt_len()
            return res

    def min_max(self) -> tuple[int, int]:
        if self.expr_alternations is None:
            return 0, 0
        min_res, max_res = self.expr_alternations[0].min_max()
        for alternation in self.expr_alternations:
            x_min, x_max = alternation.min_max()
            min_res = min(min_res, x_min)
            max_res = max(max_res, x_max)
        return min_res, max_res

    def endswith(self, end: str) -> bool:
        res = True
        for alternation in self.expr_alternations:
            if not alternation.endswith(end):
                res = False
        return res

    def remove(self, end: str):
        res = []
        for alternation in self.expr_alternations:
            if len(alternation) != 0:
                remove_len = alternation.remove(end)
                if remove_len == 0:
                    continue
            res.append(alternation)
        self.expr_alternations = res
        return self

    def return_class_repr(self) -> ClassType | None:
        alts = [
            alternation.return_class_repr()
            for alternation in self.expr_alternations
            if alternation.min_max() != (0, 0)
        ]
        if not alts:
            return None

        if any(alternation is None for alternation in alts):
            return None

        res: ClassType = alts.pop()  # type: ignore
        while alts:
            res |= alts.pop()  # type: ignore

        return res

    def get_ngrams(self, ngram_len: int) -> list[str]:
        return []

    def heuristic(self) -> Expression:
        alternations_sorted = sorted(self.expr_alternations, key=repr)

        min_res, max_res, maybe, one = self.analyze(alternations_sorted)
        if self.alt_len() == 2:
            if maybe:
                if max_res <= 2:
                    return self

            elif one:
                if all(
                    isinstance(alternation, Literal)
                    for alternation in alternations_sorted
                ):
                    for alternation in alternations_sorted:
                        if any(
                            repr(alternation).lower().endswith(y) for y in find_keywords
                        ):
                            return self

                    res_set = []
                    for alternation in alternations_sorted:
                        res_set.append(Literal(repr(alternation)[len(alternation) - 1]))
                        alternation.remove(repr(alternation)[len(alternation) - 1])
                    res = Concatenation(self, Class(res_set)).heuristic()
                    return res

        return self.simplify(min_res, max_res, maybe)

    def simplify(self, min_res: int, max_res: int, maybe: bool):
        class_set = []
        res_type = self.return_class_repr()
        if res_type is not None and res_type <= ClassType.HEX:
            class_set = get_cls_type(res_type)

        if class_set:
            return self.set_interval(class_set, min_res, max_res, maybe)
        else:
            if self.alt_len() >= 5 and res_type is not None:
                class_set = get_cls_type(res_type)
                return self.set_interval(class_set, min_res, max_res, maybe)
            else:
                self.expr_alternations = [
                    each_string.heuristic() for each_string in self.expr_alternations
                ]
                return self

    def analyze(self, alternations_sorted) -> tuple[int, int, bool, bool]:
        maybe = False
        one = False
        min_res = sys.maxsize
        max_res = 0
        for alternation in alternations_sorted:
            x_min, x_max = alternation.min_max()
            if x_min == 0:
                maybe = True
                continue
            elif x_min == 1:
                one = True
            min_res = min(min_res, x_min)
            max_res = max(max_res, x_max)
        return min_res, max_res, maybe, one

    @staticmethod
    def set_interval(
        class_set: Sequence[Expression], min_res: int, max_res: int, maybe: bool
    ):
        if min_res == 0 and maybe:
            min_res = max_res

        expr: Expression = Class(class_set)
        if min_res != 1 or max_res != 1:
            expr = Range(expr, min_res, max_res)

        if maybe:
            expr = Range(expr, 0, 1)
        return expr.heuristic()

    def contains_meta(self, meta: "Meta") -> bool:
        return any(x.contains_meta(meta) for x in self.expr_alternations)

    def reduce(self) -> Expression:
        return Literal(".*")


class Meta(Expression):
    def __init__(self, meta):
        self.meta = meta

    def __eq__(self, other) -> bool:
        return isinstance(other, Meta) and self.meta == other.meta

    def __hash__(self):
        return hash(self.meta)

    def __repr__(self) -> str:
        return self.meta

    def __str__(self) -> str:
        return self.meta

    def __len__(self) -> int:
        return len(self.meta)

    def __iter__(self):
        return iter([self])

    def remove(self, end: str):
        logger.warning(f"Remove in meta class used: {end} {self.meta}")
        return self

    def heuristic(self) -> Expression:
        return self

    def get_ngrams(self, ngram_len: int) -> list[str]:
        return []

    def return_class_repr(self) -> ClassType | None:
        if self.meta == ".":
            return ClassType.GENERAL
        return None

    def min_max(self) -> tuple[int, int]:
        return 1, 1

    def alt_len(self) -> int:
        return 1

    def endswith(self, end: str) -> bool:
        return False

    def contains_meta(self, meta: "Meta") -> bool:
        return meta == self

    def reduce(self) -> Expression:
        return Literal(".*")


class Range(Expression):
    expr: Expression

    @property
    def min(self) -> int:
        return self._min

    @min.setter
    def min(self, val: int):
        self._min = val
        self.check()

    @property
    def max(self) -> int:
        return self._max

    @max.setter
    def max(self, val: int):
        self._max = val
        self.check()

    def check(self):
        if (self.min > self.max) or (self.min == self.max and self.min == 0):
            raise ValueError(f"Invalid range ({self.min},{self.max})")

    def __init__(self, expr: Expression, min_: int, max_: int):
        self.expr = expr
        self._min = min_
        self._max = max_
        self.check()

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Range)
            and self.expr == other.expr
            and self.min == other.min
            and self.max == other.max
        )

    def __hash__(self):
        return hash((self.expr, self.min, self.max))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        expr = str(self.expr)
        if (
            isinstance(self.expr, Literal) and len(self.expr.literal) > 1
        ) or isinstance(self.expr, (Concatenation, Range)):
            expr = f"({expr})"
        if self.min == 0 and self.max == 1:
            return f"{expr}?"
        if self.max == self.min:
            if self.max == 1:
                return str(self.expr)
            return "".join([expr, "{", str(self.max), "}"])
        if self.min == 0:
            return "".join([expr, "{,", str(self.max), "}"])
        return "".join([expr, "{", str(self.min), ",", str(self.max), "}"])

    def __len__(self) -> int:
        return len(self.expr)

    def __iter__(self):
        return iter([self])

    def remove(self, end: str):
        if isinstance(self.expr, Literal):
            remove_len = len(end) // len(self.expr.literal)
            expr = Literal(self.expr.literal * remove_len).remove(end)
            return Concatenation(
                Range(self.expr, self.min - remove_len, self.max - remove_len), expr
            )

        expr = copy.deepcopy(self.expr)
        expr.remove(end)
        return Concatenation(Range(self.expr, self.min - 1, self.max - 1), expr)

    def heuristic(self) -> Expression:
        expr = self.expr.heuristic()
        return Range(expr, self.min, self.max)

    def get_ngrams(self, ngram_len: int) -> list[str]:
        if self.min == 0:
            return []
        elif self.min == 1:
            return self.expr.get_ngrams(ngram_len)
        elif isinstance(self.expr, Literal):
            return Literal(self.expr.literal * self.min).get_ngrams(ngram_len)
        elif isinstance(self.expr, Concatenation):
            expr_r = self.expr.get_right()
            expr_l = self.expr.get_left()
            if isinstance(expr_l, Literal) and isinstance(expr_r, Literal):
                return Literal(expr_r.literal + expr_l.literal).get_ngrams(ngram_len)

        return []

    def return_class_repr(self) -> ClassType | None:
        return self.expr.return_class_repr()

    def min_max(self) -> tuple[int, int]:
        expr_mn, expr_mx = self.expr.min_max()
        return expr_mn * self.min, expr_mx * self.max

    def alt_len(self) -> int:
        return self.expr.alt_len() * self.max

    def endswith(self, end: str) -> bool:
        if self.min == 0:
            return False
        if isinstance(self.expr, Literal):
            return Literal(self.expr.literal * self.min).endswith(end)
        return self.expr.endswith(end)

    def contains_meta(self, meta: "Meta") -> bool:
        return self.expr.contains_meta(meta)

    def reduce(self) -> Expression:
        return Literal(".*")


def get_cls_type(cls_type: ClassType) -> list[Expression]:
    """Create cls type representation used in heuristic"""
    if cls_type & ClassType.GENERAL:
        return [Literal(x) for x in class_types_array[ClassType.GENERAL]]
    result: set[Literal] = set()
    for i in ClassType:
        if i & cls_type:
            result.update(Literal(x) for x in class_types_array[i])
    return list(result)


def create_type(cls_type: ClassType) -> Expression:  # noqa: C901
    """Create expanded class"""
    if cls_type == ClassType.GENERAL:
        return Meta(".")

    cls_set: set[Expression] = set()
    if cls_type & ClassType.SPACE:
        cls_set.update(Literal(x) for x in string.whitespace)
    if cls_type & ClassType.NUMBERS:
        cls_set.update(Literal(x) for x in string.digits)
    if cls_type & ClassType.HEX_LOWER:
        cls_set.update(Literal(x) for x in hex_lowercase)
    if cls_type & ClassType.HEX_UPPER:
        cls_set.update(Literal(x) for x in hex_uppercase)
    if cls_type & ClassType.HEX:
        cls_set.update(Literal(x) for x in hex_lowercase + hex_uppercase)
    if cls_type & ClassType.ALPHA_LOWER:
        cls_set.update(Literal(x) for x in string.ascii_lowercase)
    if cls_type & ClassType.ALPHA_UPPER:
        cls_set.update(Literal(x) for x in string.ascii_uppercase)
    if cls_type & ClassType.ALPHA:
        cls_set.update(Literal(x) for x in string.ascii_letters)
    if cls_type & ClassType.ALPHA_NUM_LOWER:
        cls_set.update(Literal(x) for x in string.ascii_lowercase + string.digits)
    if cls_type & ClassType.ALPHA_NUM_UPPER:
        cls_set.update(Literal(x) for x in string.ascii_uppercase + string.digits)
    if cls_type & ClassType.ALPHA_NUM:
        cls_set.update(Literal(x) for x in string.ascii_uppercase + string.digits)
    if cls_type & ClassType.WORD:
        cls_set.update(Literal(x) for x in string.ascii_uppercase + string.digits + "_")

    return Class(cls_set)
