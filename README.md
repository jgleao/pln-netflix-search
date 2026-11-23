# Busca em Catálogo Netflix (Busca Ingênua, BM25 e WAND)

Projeto de Processamento de Linguagem Natural (PLN) comparando três estratégias de recuperação da informação sobre o dataset **[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)**:

1. **Busca Ingênua (`NaiveSearchEngine`)**: varredura linear força-bruta baseada em frequência simples de termos (TF bruto).
2. **BM25 Puro (`BM25SearchEngine`)**: índice invertido com ranqueamento probabilístico Okapi BM25 ($k_1 = 1{,}5$, $b = 0{,}75$) e avaliação exaustiva de candidatos.
3. **WAND (`WANDSearchEngine`)**: algoritmo *Weak AND* (Broder et al., 2003) com poda dinâmica segura (*safe dynamic pruning*) garantindo resultados Top-$K$ **100% idênticos** ao BM25 puro.

📊 **Relatório completo com gráficos e análise dos benchmarks:** veja [RELATORIO.md](RELATORIO.md).

---

## Estrutura do Repositório

```text
pln-netflix-search/
├── data/
│   └── netflix_titles.csv          # Dataset Netflix Shows (8.807 títulos)
├── notebooks/
│   └── pln_netflix_search.ipynb    # Notebook pronto para executar no Google Colab
├── results/
│   ├── benchmark_queries_k10.csv   # Métricas por consulta (K=10)
│   ├── benchmark_varying_k.csv     # Métricas variando Top-K (5, 10, 20, 50, 100)
│   ├── summary.json                # Resumo consolidado dos experimentos
│   └── figures/                    # Gráficos PNG gerados para o relatório
├── scripts/
│   ├── download_dataset.py         # Download automático do dataset no Kaggle
│   └── run_benchmarks.py           # Executa benchmarks e gera CSVs, JSON e gráficos
├── src/
│   ├── preprocessing.py            # Carregamento, normalização, tokenização e stopwords
│   ├── models.py                   # Dataclasses compartilhadas (SearchResult, SearchResponse)
│   ├── naive_search.py             # Busca Ingênua (linear)
│   ├── bm25.py                     # Índice Invertido + BM25 Puro
│   ├── wand.py                     # Algoritmo WAND sobre o índice BM25
│   └── cli.py                      # Interface de linha de comando (consulta única/interativa)
├── tests/
│   └── test_search_engines.py      # Testes unitários e de equivalência BM25 == WAND
├── RELATORIO.md                    # Relatório completo em Markdown com gráficos
└── requirements.txt                # Dependências Python
```

---

## Como Rodar Localmente

### 1. Instalar dependências

Recomenda-se Python 3.10+ em um ambiente virtual:

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Baixar o dataset (caso não esteja em `data/netflix_titles.csv`)

```bash
python scripts/download_dataset.py
```

### 3. Executar os testes unitários

Valida casos de borda e confirma a igualdade exata do Top-$K$ entre `BM25SearchEngine` e `WANDSearchEngine`:

```bash
python -m pytest -v
```

### 4. Fazer buscas pela linha de comando (CLI)

Consulta única comparando os três motores:

```bash
python -m src.cli --query "space alien adventure" --top-k 5 --engine all
```

Para filtrar apenas filmes (`Movie`) ou abrir o modo interativo:

```bash
python -m src.cli --type Movie --top-k 5
```

### 5. Reproduzir os benchmarks e gerar todos os gráficos

```bash
python scripts/run_benchmarks.py
```

Os resultados tabulares e gráficos serão atualizados em `results/` e `results/figures/`.

---

## Execução no Google Colab

Você também pode abrir e executar o notebook [`notebooks/pln_netflix_search.ipynb`](notebooks/pln_netflix_search.ipynb) diretamente no Google Colab — ele já baixa o dataset automaticamente e executa busca, benchmarks e plotagem dos gráficos na nuvem.
