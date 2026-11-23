from __future__ import annotations

import heapq
import time
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from operator import attrgetter

from .bm25 import BM25Index
from .models import SearchResponse, build_search_result
from .preprocessing import tokenize

_FLOAT_EPS = 1e-12


@dataclass(slots=True)
class _PostingCursor:
    term_order: int
    doc_ids: list[int]
    scores: list[float]
    q_tf: int
    ub: float
    length: int
    pos: int = 0
    doc_id: int = 0

    def __post_init__(self) -> None:
        self.doc_id = self.doc_ids[0]

    def advance(self) -> bool:
        self.pos += 1
        if self.pos >= self.length:
            return False
        self.doc_id = self.doc_ids[self.pos]
        return True

    def seek(self, target_doc_id: int) -> bool:
        new_pos = bisect_left(self.doc_ids, target_doc_id, self.pos)
        if new_pos >= self.length:
            self.pos = self.length
            return False
        self.pos = new_pos
        self.doc_id = self.doc_ids[new_pos]
        return True


class WANDSearchEngine:
    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, k: int = 10) -> SearchResponse:
        start_ns = time.perf_counter_ns()
        query_tokens = tokenize(query, remove_stopwords=self.index.remove_stopwords)

        if k <= 0 or not query_tokens:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return SearchResponse(
                query=query,
                algorithm="wand",
                top_k=k,
                results=[],
                latency_ms=elapsed_ms,
                docs_evaluated=0,
                postings_evaluated=0,
                seeks_performed=0,
            )

        query_counts = Counter(query_tokens)
        cursors: list[_PostingCursor] = []
        for term_order, (term, q_tf) in enumerate(sorted(query_counts.items())):
            posting = self.index.postings.get(term)
            if posting is None or not posting.doc_ids:
                continue
            cursors.append(
                _PostingCursor(
                    term_order=term_order,
                    doc_ids=posting.doc_ids,
                    scores=posting.scores,
                    q_tf=q_tf,
                    ub=posting.max_score * q_tf,
                    length=len(posting.doc_ids),
                )
            )

        if not cursors:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            return SearchResponse(
                query=query,
                algorithm="wand",
                top_k=k,
                results=[],
                latency_ms=elapsed_ms,
                docs_evaluated=0,
                postings_evaluated=0,
                seeks_performed=0,
            )

        heap: list[tuple[float, int]] = []
        docs_evaluated = 0
        postings_evaluated = 0
        seeks_performed = 0
        sort_key = attrgetter("doc_id")

        while cursors:
            cursors.sort(key=sort_key)

            if len(heap) < k:
                pivot_idx = 0
            else:
                threshold = heap[0][0]
                ub_sum = 0.0
                pivot_idx = -1
                for idx, cursor in enumerate(cursors):
                    ub_sum += cursor.ub
                    if ub_sum + _FLOAT_EPS >= threshold:
                        pivot_idx = idx
                        break
                if pivot_idx == -1:
                    break

            pivot_doc = cursors[pivot_idx].doc_id

            if cursors[0].doc_id == pivot_doc:
                matched: list[tuple[int, float]] = []
                next_cursors: list[_PostingCursor] = []
                i = 0
                n_cursors = len(cursors)
                while i < n_cursors and cursors[i].doc_id == pivot_doc:
                    c = cursors[i]
                    partial = c.scores[c.pos]
                    if c.q_tf != 1:
                        partial *= c.q_tf
                    matched.append((c.term_order, partial))
                    if c.advance():
                        next_cursors.append(c)
                    i += 1
                if i < n_cursors:
                    next_cursors.extend(cursors[i:])
                cursors = next_cursors

                if len(matched) > 2:
                    matched.sort(key=lambda item: item[0])
                doc_score = 0.0
                for _, partial in matched:
                    doc_score += partial

                docs_evaluated += 1
                postings_evaluated += len(matched)

                candidate = (doc_score, -pivot_doc)
                if len(heap) < k:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
            else:
                pick_idx = 0
                best_ub = cursors[0].ub
                for idx in range(1, pivot_idx):
                    if cursors[idx].doc_id < pivot_doc and cursors[idx].ub > best_ub:
                        best_ub = cursors[idx].ub
                        pick_idx = idx

                seeks_performed += 1
                if not cursors[pick_idx].seek(pivot_doc):
                    cursors.pop(pick_idx)

        heap.sort(reverse=True)
        results = [
            build_search_result(self.index.documents[-neg_doc_id], score)
            for score, neg_doc_id in heap
        ]
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        return SearchResponse(
            query=query,
            algorithm="wand",
            top_k=k,
            results=results,
            latency_ms=elapsed_ms,
            docs_evaluated=docs_evaluated,
            postings_evaluated=postings_evaluated,
            seeks_performed=seeks_performed,
        )
