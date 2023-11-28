import pytest

from genrex.classes import Alternation, Class, Concatenation, Literal, Meta, Range


@pytest.mark.parametrize(
    "expected, expression",
    [
        # Simple literal
        ("a?", Range(Literal("a"), 0, 1)),
        ("a{1,5}", Range(Literal("a"), 1, 5)),
        ("a{,5}", Range(Literal("a"), 0, 5)),
        ("a{5}", Range(Literal("a"), 5, 5)),
        # Longer Literal
        ("(expr)?", Range(Literal("expr"), 0, 1)),
        ("(expr){1,5}", Range(Literal("expr"), 1, 5)),
        ("(expr){,5}", Range(Literal("expr"), 0, 5)),
        ("(expr){5}", Range(Literal("expr"), 5, 5)),
        # Other expressions
        ("[abc]?", Range(Class([Literal("a"), Literal("b"), Literal("c")]), 0, 1)),
        (
            "([ab]b)?",
            Range(
                Concatenation(Class([Literal("a"), Literal("b")]), Literal("b")), 0, 1
            ),
        ),
        ("(ab|bc)?", Range(Alternation([Literal("ab"), Literal("bc")]), 0, 1)),
        ("((ab){1,2}){3,4}", Range(Range(Literal("ab"), 1, 2), 3, 4)),
        ("(ab){3,4}", Range(Range(Literal("ab"), 1, 1), 3, 4)),
        ("((ab){1,2})?", Range(Range(Literal("ab"), 1, 2), 0, 1)),
    ],
)
def test_range(expression, expected):
    assert str(expression) == expected


def test_invalid_range():
    with pytest.raises(ValueError):
        Range(Literal(""), 0, 0)
    with pytest.raises(ValueError):
        Range(Literal(""), 5, 0)


def test_equality_of_classes():
    assert Literal("abc") == Literal("abc")
    assert Class([Literal("a"), Literal("b")]) == Class([Literal("b"), Literal("a")])
    assert Concatenation(
        Class([Literal("a"), Literal("b")]), Literal("b")
    ) == Concatenation(Class([Literal("a"), Literal("b")]), Literal("b"))
    assert Alternation([Literal("ac"), Literal("ca")]) == Alternation(
        [Literal("ca"), Literal("ac")]
    )
    assert Meta("\\b") == Meta("\\b")
    assert Range(Literal("a"), 0, 1) == Range(Literal("a"), 0, 1)


def test_non_equality_of_classes():
    assert Literal("abc_") != Literal("abc")
    assert Class([Literal("c"), Literal("b")]) != Class([Literal("b")])
    assert Concatenation(
        Class([Literal("a"), Literal("b")]), Literal("bc")
    ) == Concatenation(Class([Literal("a"), Literal("b")]), Literal("bc"))
    assert Meta("\\b") != Meta("\\B")
    assert Range(Literal("b"), 0, 1) != Range(Literal("a"), 0, 1)


def test_hash_of_classes():
    assert hash(Literal("abc")) == hash(Literal("abc"))
    assert hash(Class([Literal("a"), Literal("b")])) == hash(
        Class([Literal("b"), Literal("a"), Literal("a")])
    )
    assert hash(
        Concatenation(Class([Literal("a"), Literal("b")]), Literal("bc"))
    ) == hash(Concatenation(Class([Literal("a"), Literal("b")]), Literal("bc")))
    assert hash(Alternation([Literal("a"), Literal("b")])) == hash(
        Alternation([Literal("b"), Literal("a")])
    )
    assert hash(Meta("\\b")) == hash(Meta("\\b"))
    assert hash(Range(Literal("a"), 0, 1)) == hash(Range(Literal("a"), 0, 1))


@pytest.mark.parametrize(
    ("ngram", "expr", "n_grams"),
    [
        [5, Literal("qwertyui"), ["qwert", "werty", "ertyu", "rtyui"]],
        [5, Class([Literal("a"), Literal("b")]), []],
        [
            5,
            Concatenation(Class([Literal("a"), Literal("b")]), Literal("qwertyui")),
            ["qwert", "werty", "ertyu", "rtyui"],
        ],
        [
            5,
            Concatenation(
                Literal("qwertyui"),
                Class([Literal("a"), Literal("b")]),
            ),
            ["qwert", "werty", "ertyu", "rtyui"],
        ],
        [
            5,
            Concatenation(
                Literal("qwertyui"),
                Class([Literal("a"), Literal("b")]),
            ),
            ["qwert", "werty", "ertyu", "rtyui"],
        ],
        [
            5,
            Alternation(
                [
                    Class([Literal("a"), Literal("b")]),
                    Literal("qwertyui"),
                    Literal("asdfghjkl"),
                ]
            ),
            [],
        ],
        [5, Meta("\\d"), []],
        [5, Range(Literal("qwertyui"), 0, 5), []],
        [5, Range(Literal("qwertyui"), 1, 5), ["qwert", "werty", "ertyu", "rtyui"]],
        [
            5,
            Range(Literal("qwertyui"), 3, 5),
            ["qwert", "werty", "ertyu", "rtyui", "tyuiq", "yuiqw", "uiqwe", "iqwer"],
        ],
        [
            5,
            Range(
                Concatenation(Literal("qw"), Concatenation(Meta("."), Literal("asd"))),
                1,
                3,
            ),
            [],
        ],
        [
            5,
            Range(
                Concatenation(Literal("qw"), Concatenation(Meta("."), Literal("asd"))),
                2,
                3,
            ),
            ["asdqw"],
        ],
        [6, Literal("qwertyui"), ["qwerty", "wertyu", "ertyui"]],
        [6, Class([Literal("a"), Literal("b")]), []],
        [
            6,
            Concatenation(Class([Literal("a"), Literal("b")]), Literal("qwertyui")),
            ["qwerty", "wertyu", "ertyui"],
        ],
        [
            6,
            Concatenation(
                Literal("qwertyui"),
                Class([Literal("a"), Literal("b")]),
            ),
            ["qwerty", "wertyu", "ertyui"],
        ],
        [
            6,
            Concatenation(
                Literal("qwertyui"),
                Class([Literal("a"), Literal("b")]),
            ),
            ["qwerty", "wertyu", "ertyui"],
        ],
        [
            6,
            Alternation(
                [
                    Class([Literal("a"), Literal("b")]),
                    Literal("qwertyui"),
                    Literal("asdfghjkl"),
                ]
            ),
            [],
        ],
        [6, Meta("\\d"), []],
        [6, Range(Literal("qwertyui"), 0, 5), []],
        [6, Range(Literal("qwertyui"), 1, 5), ["qwerty", "wertyu", "ertyui"]],
        [
            6,
            Range(Literal("qwertyui"), 3, 5),
            [
                "qwerty",
                "wertyu",
                "ertyui",
                "rtyuiq",
                "tyuiqw",
                "yuiqwe",
                "uiqwer",
                "iqwert",
            ],
        ],
        [
            6,
            Range(
                Concatenation(Literal("qw"), Concatenation(Meta("."), Literal("asd"))),
                1,
                3,
            ),
            [],
        ],
        [
            6,
            Range(
                Concatenation(Literal("qwe"), Concatenation(Meta("."), Literal("asd"))),
                2,
                3,
            ),
            ["asdqwe"],
        ],
    ],
)
def test_ngram_extraction(ngram, expr, n_grams):
    assert all(len(x) == ngram for x in n_grams)
    actual = expr.get_ngrams(ngram)
    assert set(actual) == set(n_grams)


if __name__ == "__main__":
    pytest.main()
