from __future__ import annotations

import argparse
from pathlib import Path

from .bm25 import BM25Index, BM25SearchEngine
from .models import SearchResponse
from .naive_search import NaiveSearchEngine
from .preprocessing import load_netflix_corpus
from .wand import WANDSearchEngine


def _print_response(resp: SearchResponse) -> None:
    print(
        f"\n=== {resp.algorithm.upper()} | query={resp.query!r} | k={resp.top_k} "
        f"| tempo={resp.latency_ms:.3f} ms | docs_avaliados={resp.docs_evaluated} "
        f"| postings={resp.postings_evaluated} | seeks={resp.seeks_performed} ==="
    )
    if not resp.results:
        print("  (nenhum resultado encontrado)")
        return
    for rank, item in enumerate(resp.results, start=1):
        print(
            f"  {rank:2d}. [{item.score:7.4f}] {item.title} "
            f"({item.media_type}, {item.release_year}) — {item.listed_in}"
        )


def main() -> None:
    default_csv = Path(__file__).resolve().parents[1] / "data" / "netflix_titles.csv"
    parser = argparse.ArgumentParser(
        description="Busca por filmes/séries da Netflix (Ingênua, BM25 e WAND)."
    )
    parser.add_argument("--csv", type=Path, default=default_csv, help="Caminho do CSV.")
    parser.add_argument(
        "--type",
        choices=["Movie", "TV Show"],
        default=None,
        help="Opcional: filtrar apenas Movie ou TV Show.",
    )
    parser.add_argument(
        "--engine",
        choices=["naive", "bm25", "wand", "all"],
        default="all",
        help="Mecanismo a executar (padrão: all).",
    )
    parser.add_argument("--query", "-q", type=str, default=None, help="Consulta única.")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Quantidade de resultados.")
    args = parser.parse_args()

    docs = load_netflix_corpus(args.csv, media_type=args.type)
    print(f"Documentos carregados: {len(docs)}")

    index = BM25Index(docs)
    print(
        f"Índice BM25 construído em {index.build_time_ms:.2f} ms "
        f"({len(index.postings)} termos únicos, avgdl={index.avgdl:.2f} tokens)."
    )

    engines = {
        "naive": NaiveSearchEngine(docs),
        "bm25": BM25SearchEngine(index),
        "wand": WANDSearchEngine(index),
    }

    selected = list(engines.keys()) if args.engine == "all" else [args.engine]

    def run_query(q: str) -> None:
        for name in selected:
            _print_response(engines[name].search(q, k=args.top_k))

    if args.query is not None:
        run_query(args.query)
        return

    print("\nModo interativo (digite 'sair' para encerrar):")
    while True:
        try:
            q = input("\nConsulta> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"sair", "exit", "quit"}:
            break
        run_query(q)


if __name__ == "__main__":
    main()
