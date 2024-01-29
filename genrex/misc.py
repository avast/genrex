cescapes: list[int | str] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    "a",
    "b",
    "t",
    "n",
    "v",
    "f",
    "r",
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]


filter_ngrams: dict[str, list[str]] = {
    "url": [
        "click.html",
        "click.php",
        "css",
        "images",
        "index.html",
        "index.php",
        "info.html",
        "info.php",
        "login.php",
        "main.html",
        "main.php",
        "profile.html",
        "profile.php",
        "tblci",
    ],
    "urldot": [
        r"click\.html",
        r"click\.php",
        r"css",
        r"images",
        r"index\.html",
        r"index\.php",
        r"info\.html",
        r"info\.php",
        r"login\.php",
        r"main\.html",
        r"main\.php",
        r"profile\.html",
        r"profile\.php",
        r"tblci",
    ],
    "others": [
        r"compatibility",
        r"framework",
        r"appdata",
        r"browser",
        r"windows",
        r"microsoft",
        r"local",
        r".exe",
        r"users\\",
        r"\r",
        r"\n",
        r"[^\\]+\\",
    ],
}


def ready_to_print(string: str) -> str:
    res = ""
    for char in string:
        if ord(char) >= 127:
            res += rf"\x{ord(char):02X}"
        elif ord(char) < 32:
            if cescapes[ord(char)] != 0:
                res += f"\\{str(cescapes[ord(char)])}"
            else:
                res += rf"\x{ord(char):02X}"
        else:
            res += char
    return res


def string2ngrams(string: str, ngram_len: int) -> list[str]:
    return [string[i : i + ngram_len] for i in range(len(string) - ngram_len + 1)]
