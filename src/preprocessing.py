from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "could",
        "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would", "you",
        "your", "yours", "yourself", "yourselves",
    }
)


@dataclass(slots=True)
class NetflixDocument:
    doc_id: int
    show_id: str
    media_type: str
    title: str
    director: str
    cast: str
    country: str
    release_year: int
    listed_in: str
    description: str
    raw_text: str
    tokens: list[str]


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    normalized = normalize_text(text)
    tokens = _TOKEN_RE.findall(normalized)
    if remove_stopwords:
        return [tok for tok in tokens if tok not in STOPWORDS]
    return tokens


def _clean_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_netflix_corpus(
    csv_path: str | Path,
    media_type: str | None = None,
    remove_stopwords: bool = True,
) -> list[NetflixDocument]:
    df = pd.read_csv(csv_path)
    if media_type is not None:
        df = df[df["type"] == media_type].reset_index(drop=True)

    documents: list[NetflixDocument] = []
    for idx, row in enumerate(df.itertuples(index=False)):
        title = _clean_str(getattr(row, "title", ""))
        director = _clean_str(getattr(row, "director", ""))
        cast = _clean_str(getattr(row, "cast", ""))
        country = _clean_str(getattr(row, "country", ""))
        listed_in = _clean_str(getattr(row, "listed_in", ""))
        description = _clean_str(getattr(row, "description", ""))
        raw_year = getattr(row, "release_year", 0)
        release_year = int(raw_year) if pd.notna(raw_year) else 0

        raw_text = " ".join(
            part for part in (title, listed_in, director, cast, country, description) if part
        )
        tokens = tokenize(raw_text, remove_stopwords=remove_stopwords)

        documents.append(
            NetflixDocument(
                doc_id=idx,
                show_id=_clean_str(getattr(row, "show_id", "")),
                media_type=_clean_str(getattr(row, "type", "")),
                title=title,
                director=director,
                cast=cast,
                country=country,
                release_year=release_year,
                listed_in=listed_in,
                description=description,
                raw_text=raw_text,
                tokens=tokens,
            )
        )
    return documents
