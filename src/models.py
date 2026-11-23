from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .preprocessing import NetflixDocument


@dataclass(slots=True)
class SearchResult:
    doc_id: int
    score: float
    show_id: str
    title: str
    media_type: str
    release_year: int
    listed_in: str
    description: str


@dataclass(slots=True)
class SearchResponse:
    query: str
    algorithm: str
    top_k: int
    results: list[SearchResult]
    latency_ms: float
    docs_evaluated: int
    postings_evaluated: int
    seeks_performed: int = 0


def build_search_result(doc: NetflixDocument, score: float) -> SearchResult:
    return SearchResult(
        doc_id=doc.doc_id,
        score=score,
        show_id=doc.show_id,
        title=doc.title,
        media_type=doc.media_type,
        release_year=doc.release_year,
        listed_in=doc.listed_in,
        description=doc.description,
    )
