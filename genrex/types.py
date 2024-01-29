import string
from enum import IntFlag

hex_lowercase: list[str] = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
]

hex_uppercase: list[str] = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
]


class ClassType(IntFlag):
    SPACE = 1
    NUMBERS = 2
    HEX_LOWER = 4
    HEX_UPPER = 8
    HEX = 16
    ALPHA_LOWER = 32
    ALPHA_UPPER = 64
    ALPHA = 128
    ALPHA_NUM_LOWER = 256
    ALPHA_NUM_UPPER = 512
    ALPHA_NUM = 1024
    WORD = 2048
    GENERAL = 4096


class_types_array: dict[ClassType, list[str]] = {
    ClassType.SPACE: ["\\s"],
    ClassType.NUMBERS: list(string.digits),
    ClassType.HEX_LOWER: hex_lowercase,
    ClassType.HEX_UPPER: hex_uppercase,
    ClassType.HEX: list(string.hexdigits),
    ClassType.ALPHA_LOWER: list(string.ascii_lowercase),
    ClassType.ALPHA_UPPER: list(string.ascii_uppercase),
    ClassType.ALPHA: list(string.ascii_letters),
    ClassType.ALPHA_NUM_LOWER: list(string.ascii_lowercase + string.digits),
    ClassType.ALPHA_NUM_UPPER: list(
        char for char in string.ascii_uppercase + string.digits
    ),
    ClassType.ALPHA_NUM: list(string.ascii_letters + string.digits),
    ClassType.WORD: list(string.ascii_letters + string.digits + "_"),
    ClassType.GENERAL: ["."],
}


def define_type(repr_string: str) -> ClassType | None:  # noqa: C901
    if len(repr_string) == 0:
        return None
    elif repr_string.isspace():
        return ClassType.SPACE
    elif repr_string.isdecimal():
        return ClassType.NUMBERS
    elif all(c in hex_lowercase for c in repr_string):
        return ClassType.HEX_LOWER
    elif all(c in hex_uppercase for c in repr_string):
        return ClassType.HEX_UPPER
    elif all(c in string.hexdigits for c in repr_string):
        return ClassType.HEX
    elif all(c in string.ascii_lowercase for c in repr_string):
        return ClassType.ALPHA_LOWER
    elif all(c in string.ascii_uppercase for c in repr_string):
        return ClassType.ALPHA_UPPER
    elif all(c in string.ascii_letters for c in repr_string):
        return ClassType.ALPHA
    elif all(c in string.ascii_lowercase + string.digits for c in repr_string):
        return ClassType.ALPHA_NUM_LOWER
    elif all(c in string.ascii_uppercase + string.digits for c in repr_string):
        return ClassType.ALPHA_NUM_UPPER
    elif all(c in string.ascii_letters + string.digits for c in repr_string):
        return ClassType.ALPHA_NUM
    elif all(c in string.ascii_letters + string.digits + "_" for c in repr_string):
        return ClassType.WORD
    return ClassType.GENERAL
