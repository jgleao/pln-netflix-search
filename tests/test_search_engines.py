"""Testes unitários e de equivalência exata entre BM25 puro e WAND."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.bm25 import BM25Index, BM25SearchEngine
from src.naive_search import NaiveSearchEngine
from src.preprocessing import NetflixDocument, load_netflix_corpus, tokenize
from src.wand import WANDSearchEngine


def _make_synthetic_corpus() -> list[NetflixDocument]:
    samples = [
        ("Alien Space Adventure", "Sci-Fi", "A crew discovers an alien ship in deep space."),
        ("Space Battle", "Action, Sci-Fi", "Space pilots defend Earth from alien invaders."),
        ("Romantic Paris", "Romance, Drama", "Two strangers fall in love in Paris."),
        ("Paris Mystery", "Mystery, Drama", "A detective investigates a crime in Paris."),
        ("Deep Sea Documentary", "Documentaries", "Exploring deep ocean life and coral reefs."),
        ("Space Space Love", "Sci-Fi, Romance", "A love story set in deep space station."),
        ("Crime City", "Crime, Thriller", "Undercover police infiltrate a crime syndicate."),
        ("Comedy Night", "Comedy", "Standup comedy special filmed live."),
    ]
    docs: list[NetflixDocument] = []
    for idx, (title, listed_in, desc) in enumerate(samples):
        raw = f"{title} {listed_in} {desc}"
        docs.append(
            NetflixDocument(
                doc_id=idx,
                show_id=f"s{idx + 1}",
                media_type="Movie",
                title=title,
                director="Test Director",
                cast="Actor One, Actor Two",
                country="United States",
                release_year=2020,
                listed_in=listed_in,
                description=desc,
                raw_text=raw,
                tokens=tokenize(raw),
            )
        )
    return docs


def test_empty_and_oov_queries() -> None:
    docs = _make_synthetic_corpus()
    index = BM25Index(docs)
    naive = NaiveSearchEngine(docs)
    bm25 = BM25SearchEngine(index)
    wand = WANDSearchEngine(index)

    for query in ("", "   ", "the and of", "xyznotfound123"):
        for engine in (naive, bm25, wand):
            resp = engine.search(query, k=5)
            assert resp.results == []


def test_wand_matches_bm25_exact_on_synthetic() -> None:
    docs = _make_synthetic_corpus()
    index = BM25Index(docs)
    bm25 = BM25SearchEngine(index)
    wand = WANDSearchEngine(index)

    queries = [
        "space",
        "alien space",
        "deep space love",
        "paris crime mystery",
        "space space alien",
        "drama comedy thriller action",
    ]
    for k in (1, 2, 3, 5, 10):
        for q in queries:
            r_bm25 = bm25.search(q, k=k)
            r_wand = wand.search(q, k=k)
            assert [r.doc_id for r in r_wand.results] == [r.doc_id for r in r_bm25.results]
            assert [pytest.approx(r.score, rel=1e-12) for r in r_wand.results] == [
                r.score for r in r_bm25.results
            ]


def test_wand_matches_bm25_exact_on_netflix_dataset() -> None:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "netflix_titles.csv"
    if not csv_path.exists():
        pytest.skip("netflix_titles.csv não encontrado")

    docs = load_netflix_corpus(csv_path)
    index = BM25Index(docs)
    bm25 = BM25SearchEngine(index)
    wand = WANDSearchEngine(index)

    queries = [
        "love",
        "serial killer",
        "world war documentary",
        "high school teenage drama comedy",
        "space alien sci fi adventure",
        "united states crime thriller action",
        "family christmas holiday music comedy",
        "indian bollywood drama romance music",
        "korean drama mystery suspense thriller",
        "true crime murder investigation police detective",
        "animated kids adventure magic friendship",
        "stand up comedy special live stage",
    ]

    for k in (5, 10, 20):
        for q in queries:
            r_bm25 = bm25.search(q, k=k)
            r_wand = wand.search(q, k=k)
            bm25_ids = [r.doc_id for r in r_bm25.results]
            wand_ids = [r.doc_id for r in r_wand.results]
            assert wand_ids == bm25_ids, f"Divergência para query={q!r}, k={k}"
            for rw, rb in zip(r_wand.results, r_bm25.results):
                assert rw.score == pytest.approx(rb.score, abs=1e-12)
            assert r_wand.docs_evaluated <= r_bm25.docs_evaluated
