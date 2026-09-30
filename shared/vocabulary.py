import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from shared.config import VOCABULARY_PATH

TOKEN_PATTERN = r"[a-z0-9]+(?:/[a-z0-9]+)*"
PARENTHETICAL = re.compile(r"\s*\(.*?\)")


@dataclass(frozen=True)
class Keyword:
    code: str
    term: str
    label: str


def _load_keywords() -> list[Keyword]:
    with VOCABULARY_PATH.open(encoding="utf-8") as file:
        return [Keyword(**item) for item in json.load(file)]


def _normalize(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return ascii_text.lower().strip()


KEYWORDS = _load_keywords()
KEYWORD_CODES = [keyword.code for keyword in KEYWORDS]
KEYWORD_FEATURES = [f"keyword_{code}" for code in KEYWORD_CODES]

_VECTORIZER = CountVectorizer(
    vocabulary=[keyword.term for keyword in KEYWORDS],
    token_pattern=TOKEN_PATTERN,
    ngram_range=(1, 2),
    binary=True,
    dtype=np.uint8,
)

_PORTUGUESE_PATTERNS = [
    (
        keyword.code,
        re.compile(rf"(?<![a-z0-9]){re.escape(_normalize(PARENTHETICAL.sub('', keyword.label)))}(?![a-z0-9])"),
    )
    for keyword in KEYWORDS
]


_CODE_BY_TERM = {keyword.term: keyword.code for keyword in KEYWORDS}
_COMPONENT_CODES = {
    keyword.code: [_CODE_BY_TERM[word] for word in keyword.term.split() if word in _CODE_BY_TERM]
    for keyword in KEYWORDS
    if " " in keyword.term
}


def _with_components(codes: set[str]) -> set[str]:
    expanded = set(codes)
    for code in codes:
        expanded.update(_COMPONENT_CODES.get(code, []))
    return expanded


def keyword_matrix(texts: Iterable[str]) -> csr_matrix:
    return _VECTORIZER.transform(texts)


def detect_codes(text: str | None) -> set[str]:
    if not text:
        return set()

    found = {KEYWORD_CODES[index] for index in keyword_matrix([text]).indices}
    normalized = _normalize(text)
    found.update(code for code, pattern in _PORTUGUESE_PATTERNS if pattern.search(normalized))
    return found


def keyword_features(codes: Iterable[str], text: str | None) -> dict[str, int]:
    active = _with_components(set(codes).intersection(KEYWORD_CODES) | detect_codes(text))
    return {f"keyword_{code}": int(code in active) for code in KEYWORD_CODES}
