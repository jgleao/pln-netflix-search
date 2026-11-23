from __future__ import annotations

import heapq
import time
from collections import Counter

from .models import SearchResponse, build_search_result
from .preprocessing import NetflixDocument, tokenize


class NaiveSearchEngine:
    def __init__(
        self,
        documents: list[NetflixDocument],
        remove_stopwords: bool = True,
    ) -> None:
        self.documents = documents
        self.remove_stopwords = remove_stopwords

    def search(self, query: str, k: int = 10) -> SearchResponse:
        start_ns = time.perf_counter_ns()
        query_tokens = tokenize(query, remove_stopwords=self.remove_stopwords)

        if k <= 0 or not query_tokens:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return SearchResponse(
                query=query,
                algorithm="naive",
                top_k=k,
                results=[],
                latency_ms=elapsed_ms,
                docs_evaluated=0,
                postings_evaluated=0,
            )

        query_counts = Counter(query_tokens)
        query_term_set = set(query_counts)
        scored: list[tuple[float, int]] = []
        tokens_scanned = 0

        for doc in self.documents:
            score = 0
            doc_tokens = doc.tokens
            tokens_scanned += len(doc_tokens)
            for tok in doc_tokens:
                if tok in query_term_set:
                    score += query_counts[tok]
            if score > 0:
                scored.append((float(score), -doc.doc_id))

        top_items = heapq.nlargest(k, scored)
        results = [
            build_search_result(self.documents[-neg_doc_id], score)
            for score, neg_doc_id in top_items
        ]
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return SearchResponse(
            query=query,
            algorithm="naive",
            top_k=k,
            results=results,
            latency_ms=elapsed_ms,
            docs_evaluated=len(self.documents),
            postings_evaluated=tokens_scanned,
        )
