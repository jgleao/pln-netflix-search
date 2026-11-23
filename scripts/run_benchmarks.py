from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bm25 import BM25Index, BM25SearchEngine
from src.naive_search import NaiveSearchEngine
from src.preprocessing import NetflixDocument, load_netflix_corpus
from src.wand import WANDSearchEngine

BENCHMARK_QUERIES: list[tuple[int, str]] = [
    (1, "alien"),
    (1, "love"),
    (1, "thriller"),
    (1, "documentary"),
    (1, "christmas"),
    (2, "serial killer"),
    (2, "world war"),
    (2, "standup comedy"),
    (2, "time travel"),
    (2, "high school"),
    (3, "space alien adventure"),
    (3, "true crime murder"),
    (3, "korean drama romance"),
    (3, "family christmas holiday"),
    (3, "action thriller crime"),
    (4, "indian bollywood drama romance"),
    (4, "teenage high school comedy"),
    (4, "police detective murder investigation"),
    (4, "animated kids adventure magic"),
    (4, "united states documentary music"),
    (5, "international drama family love story"),
    (5, "crime thriller mystery suspense dark"),
    (5, "comedy stand up live special"),
    (5, "war history documentary military battle"),
    (5, "sci fi space future alien"),
    (6, "united states comedy drama romance family"),
    (6, "young woman life love family friends"),
    (6, "crime police detective investigation mystery thriller"),
    (8, "united states international drama comedy action crime thriller"),
    (8, "love life family friends young man woman story"),
]


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct))


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def _measure_query(
    engine: NaiveSearchEngine | BM25SearchEngine | WANDSearchEngine,
    query: str,
    k: int,
    repeats: int,
) -> tuple[float, int, int, int, list[int]]:
    resp = engine.search(query, k=k)
    latencies = []
    for _ in range(repeats):
        r = engine.search(query, k=k)
        latencies.append(r.latency_ms)
    median_ms = statistics.median(latencies)
    doc_ids = [item.doc_id for item in resp.results]
    return median_ms, resp.docs_evaluated, resp.postings_evaluated, resp.seeks_performed, doc_ids


def plot_dataset_overview(docs: list[NetflixDocument], out_dir: Path) -> dict[str, float]:
    movies = sum(1 for d in docs if d.media_type == "Movie")
    shows = sum(1 for d in docs if d.media_type == "TV Show")
    lengths = [len(d.tokens) for d in docs]
    avgdl = float(np.mean(lengths))
    meddl = float(np.median(lengths))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    bars = ax1.bar(["Movie", "TV Show"], [movies, shows], color=["#E50914", "#221F1F"], width=0.55)
    ax1.set_title("Distribuição do Catálogo por Tipo", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Quantidade de Títulos")
    _style_axes(ax1)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 120, f"{int(h):,}", ha="center", fontweight="bold")

    ax2.hist(lengths, bins=35, color="#B20710", alpha=0.85, edgecolor="white")
    ax2.axvline(avgdl, color="#111111", linestyle="--", linewidth=1.8, label=f"Média = {avgdl:.1f} tokens")
    ax2.axvline(meddl, color="#1F77B4", linestyle=":", linewidth=1.8, label=f"Mediana = {meddl:.1f} tokens")
    ax2.set_title("Distribuição do Comprimento dos Documentos (Tokens)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Tokens após pré-processamento")
    ax2.set_ylabel("Número de Documentos")
    ax2.legend()
    _style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_dir / "01_dataset_overview.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "total_docs": len(docs),
        "movies": movies,
        "tv_shows": shows,
        "avg_tokens": avgdl,
        "median_tokens": meddl,
        "min_tokens": int(min(lengths)),
        "max_tokens": int(max(lengths)),
    }


def plot_naive_vs_bm25_latency(df_exp1: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

    metrics = ["Média", "Mediana", "P95"]
    naive_vals = [
        df_exp1["naive_ms"].mean(),
        df_exp1["naive_ms"].median(),
        _percentile(df_exp1["naive_ms"].tolist(), 95),
    ]
    bm25_vals = [
        df_exp1["bm25_ms"].mean(),
        df_exp1["bm25_ms"].median(),
        _percentile(df_exp1["bm25_ms"].tolist(), 95),
    ]

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax1.bar(x - w / 2, naive_vals, w, label="Busca Ingênua (Linear)", color="#7F8C8D")
    b2 = ax1.bar(x + w / 2, bm25_vals, w, label="BM25 Puro (Índice Invertido)", color="#E50914")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Latência (ms)")
    ax1.set_title("Latência por Consulta: Ingênua vs BM25 (K=10)", fontsize=11.5, fontweight="bold")
    ax1.legend()
    _style_axes(ax1)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.2f}", ha="center", fontsize=9)

    speedup_mean = naive_vals[0] / bm25_vals[0] if bm25_vals[0] > 0 else 0.0
    ax1.annotate(
        f"Speedup médio: {speedup_mean:.1f}x",
        xy=(0 + w / 2, bm25_vals[0]),
        xytext=(0.45, max(naive_vals) * 0.72),
        arrowprops={"arrowstyle": "->", "color": "#111111"},
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#FFF3CD", "ec": "#E0A800"},
    )

    grouped = df_exp1.groupby("q_len")[["naive_ms", "bm25_ms"]].mean().reset_index()
    ax2.plot(grouped["q_len"], grouped["naive_ms"], marker="o", linewidth=2.2, color="#7F8C8D", label="Busca Ingênua")
    ax2.plot(grouped["q_len"], grouped["bm25_ms"], marker="s", linewidth=2.2, color="#E50914", label="BM25 Puro")
    ax2.set_xlabel("Tamanho da Consulta (termos)")
    ax2.set_ylabel("Latência Média (ms)")
    ax2.set_title("Latência Média por Tamanho da Consulta", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(grouped["q_len"])
    ax2.legend()
    _style_axes(ax2)

    fig.tight_layout()
    fig.savefig(out_dir / "02_naive_vs_bm25_latency.png", bbox_inches="tight")
    plt.close(fig)


def plot_naive_vs_bm25_quality(
    avgdl: float,
    mean_len_naive: float,
    mean_len_bm25: float,
    overlaps_by_k: dict[int, float],
    out_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)

    labels = ["Média da Coleção\n(avgdl)", "Top-10\nBusca Ingênua", "Top-10\nBM25"]
    vals = [avgdl, mean_len_naive, mean_len_bm25]
    colors = ["#2C3E50", "#7F8C8D", "#E50914"]
    bars = ax1.bar(labels, vals, color=colors, width=0.5)
    ax1.set_ylabel("Comprimento Médio (tokens)")
    ax1.set_title("Viés por Documentos Longos no Top-10", fontsize=11.5, fontweight="bold")
    _style_axes(ax1)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", fontweight="bold")

    ks = sorted(overlaps_by_k.keys())
    ov_vals = [overlaps_by_k[k] * 100.0 for k in ks]
    bars2 = ax2.bar([f"K={k}" for k in ks], ov_vals, color="#1F77B4", width=0.45)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Sobreposição Média (%)")
    ax2.set_title("Sobreposição de Resultados: Ingênua vs BM25", fontsize=11.5, fontweight="bold")
    _style_axes(ax2)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.1f}%", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "03_naive_vs_bm25_quality.png", bbox_inches="tight")
    plt.close(fig)


def plot_bm25_vs_wand_pruning(
    df_exp1: pd.DataFrame,
    df_k: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

    multi = df_exp1[df_exp1["q_len"] >= 2]
    grouped = multi.groupby("q_len")[["bm25_docs", "wand_docs"]].mean().reset_index()

    x = np.arange(len(grouped))
    w = 0.36
    ax1.bar(x - w / 2, grouped["bm25_docs"], w, label="BM25 Puro (Exaustivo)", color="#221F1F")
    ax1.bar(x + w / 2, grouped["wand_docs"], w, label="WAND (Poda Dinâmica)", color="#27AE60")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(q)} termos" for q in grouped["q_len"]])
    ax1.set_ylabel("Documentos Avaliados (média)")
    ax1.set_title("Documentos Avaliados por Tamanho da Consulta (K=10)", fontsize=11.5, fontweight="bold")
    ax1.legend()
    _style_axes(ax1)

    for idx, row in grouped.iterrows():
        prune_pct = (1.0 - row["wand_docs"] / row["bm25_docs"]) * 100.0
        ax1.text(
            idx + w / 2 + 0.04,
            row["wand_docs"] + 110,
            f"-{prune_pct:.0f}%",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1E8449",
        )
    ax1.set_ylim(top=grouped["bm25_docs"].max() * 1.12)

    pruning_by_k = df_k.groupby("k")["pruning_pct"].mean().reset_index()
    bars_k = ax2.bar(
        [f"K={int(k)}" for k in pruning_by_k["k"]],
        pruning_by_k["pruning_pct"],
        color="#27AE60",
        width=0.5,
    )
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Taxa de Poda Média (%)")
    ax2.set_title("Taxa de Poda do WAND Variando Top-K (Consultas ≥ 3 termos)", fontsize=11.5, fontweight="bold")
    _style_axes(ax2)
    for bar in bars_k:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "04_bm25_vs_wand_pruning.png", bbox_inches="tight")
    plt.close(fig)


def plot_bm25_vs_wand_latency(
    df_exp1: pd.DataFrame,
    df_k: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

    multi = df_exp1[df_exp1["q_len"] >= 2]
    grouped = multi.groupby("q_len")[["bm25_ms", "wand_ms"]].mean().reset_index()

    ax1.plot(grouped["q_len"], grouped["bm25_ms"], marker="o", linewidth=2.2, color="#221F1F", label="BM25 Puro")
    ax1.plot(grouped["q_len"], grouped["wand_ms"], marker="s", linewidth=2.2, color="#27AE60", label="WAND")
    ax1.set_xlabel("Tamanho da Consulta (termos)")
    ax1.set_ylabel("Latência Média (ms)")
    ax1.set_title("Latência Média: BM25 Puro vs WAND (K=10)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(grouped["q_len"])
    ax1.legend()
    _style_axes(ax1)

    lat_by_k = df_k.groupby("k")[["bm25_ms", "wand_ms"]].mean().reset_index()
    x = np.arange(len(lat_by_k))
    w = 0.36
    ax2.bar(x - w / 2, lat_by_k["bm25_ms"], w, label="BM25 Puro", color="#221F1F")
    ax2.bar(x + w / 2, lat_by_k["wand_ms"], w, label="WAND", color="#27AE60")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={int(k)}" for k in lat_by_k["k"]])
    ax2.set_ylabel("Latência Média (ms)")
    ax2.set_title("Latência Média Variando Top-K (Consultas ≥ 3 termos)", fontsize=11.5, fontweight="bold")
    ax2.legend()
    _style_axes(ax2)

    for idx, row in lat_by_k.iterrows():
        sp = row["bm25_ms"] / row["wand_ms"] if row["wand_ms"] > 0 else 0.0
        top_h = max(row["bm25_ms"], row["wand_ms"])
        ax2.text(idx, top_h + 0.06, f"{sp:.2f}x", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(top=lat_by_k[["bm25_ms", "wand_ms"]].to_numpy().max() * 1.16)

    fig.tight_layout()
    fig.savefig(out_dir / "05_bm25_vs_wand_latency.png", bbox_inches="tight")
    plt.close(fig)


def plot_overall_comparison(df_exp1: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)

    sub = df_exp1[df_exp1["q_len"] >= 3]
    names = ["Busca Ingênua\n(Linear)", "BM25 Puro\n(Índice Invertido)", "WAND\n(Poda Dinâmica)"]
    means_ms = [sub["naive_ms"].mean(), sub["bm25_ms"].mean(), sub["wand_ms"].mean()]
    docs_eval = [sub["naive_docs"].mean(), sub["bm25_docs"].mean(), sub["wand_docs"].mean()]
    colors = ["#7F8C8D", "#E50914", "#27AE60"]

    b1 = ax1.bar(names, means_ms, color=colors, width=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Latência Média (ms, escala log)")
    ax1.set_title("Comparação Geral de Latência (Consultas ≥ 3 termos)", fontsize=11.5, fontweight="bold")
    _style_axes(ax1)
    for bar in b1:
        h = bar.get_height()
        qps = 1000.0 / h if h > 0 else 0.0
        ax1.text(bar.get_x() + bar.get_width() / 2, h * 1.15, f"{h:.2f} ms\n({qps:.0f} QPS)", ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(top=max(means_ms) * 2.8)

    b2 = ax2.bar(names, docs_eval, color=colors, width=0.5)
    ax2.set_ylabel("Documentos Avaliados / Consulta")
    ax2.set_title("Custo Algorítmico (Documentos Avaliados, K=10)", fontsize=11.5, fontweight="bold")
    _style_axes(ax2)
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 150, f"{h:,.0f}", ha="center", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(top=max(docs_eval) * 1.15)

    fig.tight_layout()
    fig.savefig(out_dir / "06_overall_comparison.png", bbox_inches="tight")
    plt.close(fig)


def run_all() -> None:
    csv_path = ROOT_DIR / "data" / "netflix_titles.csv"
    results_dir = ROOT_DIR / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    docs = load_netflix_corpus(csv_path)
    corpus_stats = plot_dataset_overview(docs, figures_dir)

    naive = NaiveSearchEngine(docs)
    index = BM25Index(docs)
    bm25 = BM25SearchEngine(index)
    wand = WANDSearchEngine(index)

    rows_exp1 = []
    naive_top_lengths: list[int] = []
    bm25_top_lengths: list[int] = []
    exact_matches = 0

    for q_len, query in BENCHMARK_QUERIES:
        n_ms, n_docs, n_post, _, n_ids = _measure_query(naive, query, k=10, repeats=5)
        b_ms, b_docs, b_post, _, b_ids = _measure_query(bm25, query, k=10, repeats=15)
        w_ms, w_docs, w_post, w_seeks, w_ids = _measure_query(wand, query, k=10, repeats=15)

        if w_ids == b_ids:
            exact_matches += 1
        else:
            raise AssertionError(f"Divergência WAND vs BM25 na consulta {query!r}")

        naive_top_lengths.extend(len(docs[d].tokens) for d in n_ids)
        bm25_top_lengths.extend(len(docs[d].tokens) for d in b_ids)

        pruning_pct = (1.0 - (w_docs / b_docs)) * 100.0 if b_docs > 0 else 0.0
        rows_exp1.append(
            {
                "q_len": q_len,
                "query": query,
                "naive_ms": n_ms,
                "naive_docs": n_docs,
                "naive_postings": n_post,
                "bm25_ms": b_ms,
                "bm25_docs": b_docs,
                "bm25_postings": b_post,
                "wand_ms": w_ms,
                "wand_docs": w_docs,
                "wand_postings": w_post,
                "wand_seeks": w_seeks,
                "pruning_pct": pruning_pct,
                "speedup_naive_vs_bm25": n_ms / b_ms if b_ms > 0 else 0.0,
                "speedup_bm25_vs_wand": b_ms / w_ms if w_ms > 0 else 0.0,
            }
        )

    df_exp1 = pd.DataFrame(rows_exp1)
    df_exp1.to_csv(results_dir / "benchmark_queries_k10.csv", index=False)

    overlaps_by_k: dict[int, float] = {}
    for k in (5, 10, 20):
        ov_scores = []
        for _, query in BENCHMARK_QUERIES:
            n_set = {r.doc_id for r in naive.search(query, k=k).results}
            b_set = {r.doc_id for r in bm25.search(query, k=k).results}
            denom = max(len(n_set | b_set), 1)
            ov_scores.append(len(n_set & b_set) / denom)
        overlaps_by_k[k] = float(np.mean(ov_scores))

    rows_k = []
    multi_queries = [(q_len, q) for q_len, q in BENCHMARK_QUERIES if q_len >= 3]
    for k in (5, 10, 20, 50, 100):
        for q_len, query in multi_queries:
            b_ms, b_docs, b_post, _, b_ids = _measure_query(bm25, query, k=k, repeats=15)
            w_ms, w_docs, w_post, w_seeks, w_ids = _measure_query(wand, query, k=k, repeats=15)
            assert w_ids == b_ids, f"Divergência WAND vs BM25 em k={k}, query={query!r}"
            pruning_pct = (1.0 - (w_docs / b_docs)) * 100.0 if b_docs > 0 else 0.0
            rows_k.append(
                {
                    "k": k,
                    "q_len": q_len,
                    "query": query,
                    "bm25_ms": b_ms,
                    "wand_ms": w_ms,
                    "bm25_docs": b_docs,
                    "wand_docs": w_docs,
                    "wand_seeks": w_seeks,
                    "pruning_pct": pruning_pct,
                    "speedup": b_ms / w_ms if w_ms > 0 else 0.0,
                }
            )

    df_k = pd.DataFrame(rows_k)
    df_k.to_csv(results_dir / "benchmark_varying_k.csv", index=False)

    qualitative_queries = ["love story", "serial killer", "space alien adventure"]
    qualitative_examples = {}
    for q in qualitative_queries:
        n_res = naive.search(q, k=5).results
        b_res = bm25.search(q, k=5).results
        w_res = wand.search(q, k=5).results
        qualitative_examples[q] = {
            "naive": [
                {"title": r.title, "type": r.media_type, "score": round(r.score, 4), "tokens": len(docs[r.doc_id].tokens)}
                for r in n_res
            ],
            "bm25": [
                {"title": r.title, "type": r.media_type, "score": round(r.score, 4), "tokens": len(docs[r.doc_id].tokens)}
                for r in b_res
            ],
            "wand": [
                {"title": r.title, "type": r.media_type, "score": round(r.score, 4), "tokens": len(docs[r.doc_id].tokens)}
                for r in w_res
            ],
        }

    plot_naive_vs_bm25_latency(df_exp1, figures_dir)
    plot_naive_vs_bm25_quality(
        corpus_stats["avg_tokens"],
        float(np.mean(naive_top_lengths)),
        float(np.mean(bm25_top_lengths)),
        overlaps_by_k,
        figures_dir,
    )
    plot_bm25_vs_wand_pruning(df_exp1, df_k, figures_dir)
    plot_bm25_vs_wand_latency(df_exp1, df_k, figures_dir)
    plot_overall_comparison(df_exp1, figures_dir)

    multi_exp1 = df_exp1[df_exp1["q_len"] >= 3]
    summary = {
        "corpus": {
            **corpus_stats,
            "vocab_size": len(index.postings),
            "index_build_ms": round(index.build_time_ms, 2),
        },
        "exact_wand_matches": f"{exact_matches}/{len(BENCHMARK_QUERIES)}",
        "exp1_all_30_queries_k10": {
            "naive_mean_ms": round(float(df_exp1["naive_ms"].mean()), 3),
            "naive_median_ms": round(float(df_exp1["naive_ms"].median()), 3),
            "naive_p95_ms": round(_percentile(df_exp1["naive_ms"].tolist(), 95), 3),
            "bm25_mean_ms": round(float(df_exp1["bm25_ms"].mean()), 3),
            "bm25_median_ms": round(float(df_exp1["bm25_ms"].median()), 3),
            "bm25_p95_ms": round(_percentile(df_exp1["bm25_ms"].tolist(), 95), 3),
            "speedup_naive_to_bm25_mean": round(
                float(df_exp1["naive_ms"].mean() / df_exp1["bm25_ms"].mean()), 2
            ),
            "mean_top10_len_naive": round(float(np.mean(naive_top_lengths)), 2),
            "mean_top10_len_bm25": round(float(np.mean(bm25_top_lengths)), 2),
            "jaccard_overlap_k": {str(k): round(v * 100, 1) for k, v in overlaps_by_k.items()},
        },
        "exp2_multi_term_k10": {
            "bm25_mean_ms": round(float(multi_exp1["bm25_ms"].mean()), 3),
            "wand_mean_ms": round(float(multi_exp1["wand_ms"].mean()), 3),
            "speedup_bm25_to_wand": round(
                float(multi_exp1["bm25_ms"].mean() / multi_exp1["wand_ms"].mean()), 2
            ),
            "bm25_mean_docs": round(float(multi_exp1["bm25_docs"].mean()), 1),
            "wand_mean_docs": round(float(multi_exp1["wand_docs"].mean()), 1),
            "mean_pruning_pct": round(float(multi_exp1["pruning_pct"].mean()), 1),
            "naive_mean_ms_multi": round(float(multi_exp1["naive_ms"].mean()), 3),
        },
        "exp2_by_query_length": df_exp1.groupby("q_len")[
            ["naive_ms", "bm25_ms", "wand_ms", "bm25_docs", "wand_docs", "pruning_pct", "wand_seeks"]
        ]
        .mean()
        .round(3)
        .to_dict(orient="index"),
        "exp2_varying_k": df_k.groupby("k")[
            ["bm25_ms", "wand_ms", "bm25_docs", "wand_docs", "pruning_pct", "speedup"]
        ]
        .mean()
        .round(3)
        .to_dict(orient="index"),
        "qualitative_examples": qualitative_examples,
    }

    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_all()
