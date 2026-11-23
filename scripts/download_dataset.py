from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/shivamb/netflix-shows"
)


def download_netflix_dataset(output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "netflix_titles.csv"
    zip_path = output_dir / "netflix-shows.zip"

    if csv_path.exists() and not force:
        print(f"Dataset já existe em: {csv_path}")
        return csv_path

    print(f"Baixando dataset de {KAGGLE_DATASET_URL} ...")
    req = urllib.request.Request(
        KAGGLE_DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp, zip_path.open("wb") as out:
        out.write(resp.read())

    print(f"Extraindo {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo esperado não encontrado após extração: {csv_path}")

    print(f"Dataset pronto em: {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download do dataset Netflix Shows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Diretório de destino para o CSV (padrão: ./data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força novo download mesmo se netflix_titles.csv já existir.",
    )
    args = parser.parse_args()
    download_netflix_dataset(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
