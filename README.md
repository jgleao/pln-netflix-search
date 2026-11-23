# Trabalho de PLN — Mecanismo de Busca no Catálogo da Netflix (Busca Ingênua, BM25 e WAND)

Olá, professor!

Desenvolvi este repositório para o trabalho da disciplina de **Processamento de Linguagem Natural (PLN)** utilizando o dataset [Netflix Movies and TV Shows (`netflix_titles.csv`)](https://www.kaggle.com/datasets/shivamb/netflix-shows). O objetivo do projeto é implementar, testar e comparar na prática três abordagens de recuperação da informação:

1. **Busca Ingênua** — varredura linear documento a documento baseada na contagem simples de termos (TF bruto), sem índice invertido.
2. **BM25 Puro** — índice invertido com ranqueamento Okapi BM25 ($k_1 = 1{,}5$, $b = 0{,}75$) e avaliação exaustiva de todos os candidatos.
3. **WAND (*Weak AND*)** — algoritmo de poda dinâmica segura (*safe dynamic pruning*) sobre o índice do BM25, garantindo exatamente o mesmo ranking Top-$K$ do BM25 puro enquanto pula documentos que não conseguem alcançar o limiar atual.

Todo o relatório comparativo com tabelas e gráficos está disponível em **[RELATORIO.md](RELATORIO.md)**. Caso prefira executar diretamente no navegador sem configurar ambiente local, também deixei um notebook pronto em **[notebooks/pln_netflix_search.ipynb](notebooks/pln_netflix_search.ipynb)** para rodar no Google Colab.

---

## Estrutura do Projeto

```text
pln-netflix-search/
├── data/
│   └── netflix_titles.csv          # Base de dados do Kaggle (8.807 títulos)
├── notebooks/
│   └── pln_netflix_search.ipynb    # Versão auto-contida para Google Colab
├── results/
│   ├── benchmark_queries_k10.csv   # Métricas por consulta (K=10)
│   ├── benchmark_varying_k.csv     # Métricas variando K (5, 10, 20, 50, 100)
│   ├── summary.json                # Resumo consolidado em JSON
│   └── figures/                    # Gráficos PNG usados no relatório
├── scripts/
│   ├── download_dataset.py         # Baixa e extrai o dataset do Kaggle
│   └── run_benchmarks.py           # Roda os benchmarks e gera os gráficos
├── src/
│   ├── preprocessing.py            # Leitura do CSV, normalização e tokenização
│   ├── models.py                   # Estruturas de resposta e resultados
│   ├── naive_search.py             # Busca Ingênua (linear)
│   ├── bm25.py                     # Índice Invertido + BM25 Puro
│   ├── wand.py                     # Algoritmo WAND
│   └── cli.py                      # Execução de buscas pelo terminal
├── tests/
│   └── test_search_engines.py      # Testes de exatidão (BM25 == WAND)
├── RELATORIO.md                    # Relatório completo com análise e gráficos
└── requirements.txt                # Dependências Python
```

---

## Passo a Passo para Replicar em Outra Máquina

### Pré-requisitos
- **Python 3.10 ou superior** instalado com `pip`.
- Acesso à internet apenas para instalar as bibliotecas (e para baixar o CSV, caso ainda não esteja na pasta `data/`).

---

### 1. Clonar (ou baixar) o repositório e entrar na pasta

```bash
git clone https://github.com/jgleao/pln-netflix-search.git
cd pln-netflix-search
```

---

### 2. Criar e ativar um ambiente virtual Python

**No Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Se o PowerShell bloquear a execução de scripts na sua máquina, rode antes `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente ativar novamente.)*

**No Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

As únicas bibliotecas externas usadas são `pandas`, `numpy`, `matplotlib` e `pytest`.

---

### 4. Conferir ou baixar o dataset

O arquivo `data/netflix_titles.csv` já acompanha o repositório. Se ele não estiver presente ou você quiser baixá-lo novamente do Kaggle, basta executar:

```bash
python scripts/download_dataset.py
```

*(Caso o download automático falhe por algum bloqueio de rede, basta baixar o arquivo no [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows), extrair o `netflix_titles.csv` e colocá-lo dentro da pasta `data/`.)*

---

### 5. Rodar os testes automatizados

Este passo valida os casos de borda e comprova que o **WAND retorna exatamente os mesmos resultados e pontuações do BM25 puro**:

```bash
python -m pytest -v
```

Saída esperada: `3 passed`.

---

### 6. Testar buscas manualmente no terminal (CLI)

Para comparar os três mecanismos lado a lado em uma consulta específica:

```bash
python -m src.cli --query "space alien adventure" --top-k 5 --engine all
```

Outros modos úteis:
- **Modo interativo** (permite digitar várias consultas em sequência):
  ```bash
  python -m src.cli --top-k 5
  ```
- **Filtrar apenas filmes (`Movie`) ou apenas séries (`TV Show`)**:
  ```bash
  python -m src.cli --type Movie --query "serial killer" --top-k 5
  ```
- **Rodar apenas um algoritmo (`naive`, `bm25` ou `wand`)**:
  ```bash
  python -m src.cli --engine wand --query "world war documentary" --top-k 10
  ```

---

### 7. Reproduzir todos os benchmarks e gerar os gráficos novamente

Para reexecutar todas as medições (30 consultas em `K=10` + variação de `K` em `5, 10, 20, 50, 100`) na sua máquina e regenerar os arquivos em `results/` e `results/figures/`:

```bash
python scripts/run_benchmarks.py
```

Ao terminar, os 6 gráficos PNG estarão atualizados em `results/figures/` e os resultados numéricos da sua máquina ficarão salvos em `results/summary.json`, `results/benchmark_queries_k10.csv` e `results/benchmark_varying_k.csv`.

---

## Executando no Google Colab (Alternativa sem instalação local)

1. Abra diretamente no Google Colab pelo link: [Abrir notebook no Colab](https://colab.research.google.com/github/jgleao/pln-netflix-search/blob/master/notebooks/pln_netflix_search.ipynb) (ou faça upload de [notebooks/pln_netflix_search.ipynb](notebooks/pln_netflix_search.ipynb)).
2. Execute as células em ordem (`Runtime -> Run all`). O notebook já baixa o dataset do Kaggle, constrói os índices, valida `BM25 == WAND` e plota os gráficos comparativos.
