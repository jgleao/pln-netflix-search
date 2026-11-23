"""Índice invertido e mecanismo de busca Okapi BM25 puro (avaliação exaustiva)."""

from __future__ import annotations

import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass

from .models import SearchResponse, build_search_result
from .preprocessing import NetflixDocument, tokenize


@dataclass(slots=True)
class PostingList:
    """Lista invertida ordenada por doc_id com scores BM25 parciais pré-computados."""

    term: str
    doc_ids: list[int]
    scores: list[float]
    max_score: float
    idf: float
    df: int


class BM25Index:
    """Índice invertido com estatísticas da coleção e upper bounds para BM25/WAND."""

    def __init__(
        self,
        documents: list[NetflixDocument],
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.num_docs = len(documents)
        self.doc_lengths = [len(doc.tokens) for doc in documents]
        self.avgdl = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0.0
        )
        self.postings: dict[str, PostingList] = {}
        self.build_time_ms: float = 0.0
        self._build_index()

    def _build_index(self) -> None:
        start_ns = time.perf_counter_ns()
        raw_postings: dict[str, list[tuple[int, int]]] = {}

        for doc in self.documents:
            if not doc.tokens:
                continue
            tf_map = Counter(doc.tokens)
            for term, tf in tf_map.items():
                bucket = raw_postings.get(term)
                if bucket is None:
                    raw_postings[term] = [(doc.doc_id, tf)]
                else:
                    bucket.append((doc.doc_id, tf))

        avgdl = self.avgdl if self.avgdl > 0.0 else 1.0
        n_docs = self.num_docs
        k1 = self.k1
        b = self.b

        for term, entries in raw_postings.items():
            df = len(entries)
            # IDF estilo Lucene: estritamente positivo para todo termo (requisito do WAND).
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            doc_ids: list[int] = []
            scores: list[float] = []
            max_score = 0.0

            for doc_id, tf in entries:
                dl = self.doc_lengths[doc_id]
                norm = k1 * (1.0 - b + b * (dl / avgdl))
                score = idf * (tf * (k1 + 1.0)) / (tf + norm)
                doc_ids.append(doc_id)
                scores.append(score)
                if score > max_score:
                    max_score = score

            self.postings[term] = PostingList(
                term=term,
                doc_ids=doc_ids,
                scores=scores,
                max_score=max_score,
                idf=idf,
                df=df,
            )

        self.build_time_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0


class BM25SearchEngine:
    """Busca BM25 exaustiva sobre o índice invertido (sem poda dinâmica)."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, k: int = 10) -> SearchResponse:
        """Avalia todos os documentos candidatos da consulta e retorna o Top-K."""
        start_ns = time.perf_counter_ns()
        query_tokens = tokenize(query, remove_stopwords=self.index.remove_stopwords)

        if k <= 0 or not query_tokens:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return SearchResponse(
                query=query,
                algorithm="bm25",
                top_k=k,
                results=[],
                latency_ms=elapsed_ms,
                docs_evaluated=0,
                postings_evaluated=0,
            )

        query_counts = Counter(query_tokens)
        doc_scores: dict[int, float] = {}
        postings_evaluated = 0

        for term, q_tf in sorted(query_counts.items()):
            posting = self.index.postings.get(term)
            if posting is None:
                continue
            doc_ids = posting.doc_ids
            scores = posting.scores
            postings_evaluated += len(doc_ids)
            if q_tf == 1:
                for doc_id, s in zip(doc_ids, scores):
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + s
            else:
                for doc_id, s in zip(doc_ids, scores):
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + s * q_tf

        top_items = heapq.nlargest(
            k,
            ((score, -doc_id) for doc_id, score in doc_scores.items()),
        )
        results = [
            build_search_result(self.index.documents[-neg_doc_id], score)
            for score, neg_doc_id in top_items
        ]
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return SearchResponse(
            query=query,
            algorithm="bm25",
            top_k=k,
            results=results,
            latency_ms=elapsed_ms,
            docs_evaluated=len(doc_scores),
            postings_evaluated=postings_evaluated,
        )
