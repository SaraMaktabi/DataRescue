"""
Adapte le dataset Kaggle 'transactions-fraud-datasets' au schéma du pipeline DataRescue.

Entrées  : data/raw/kaggle/transactions_data.csv, cards_data.csv, users_data.csv
Sorties  : data/raw/transactions.csv  (janvier 2010, colonnes nettoyées)
           data/raw/cards.csv         (copie brute)
           data/raw/users.csv         (copie brute)

Usage :
    uv run python -m src.data_scripts.adapt_kaggle
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config_loader import CONFIG
from src.utils.logging import get_logger

log = get_logger("adapt_kaggle")

# Chemins
RAW_KAGGLE = Path("data/raw/kaggle")
RAW_OUT = Path("data/raw")

# Période à extraire (depuis config, avec valeurs par défaut)
DATE_START = CONFIG.get("data", {}).get("kaggle_date_start", "2010-01-01")
DATE_END = CONFIG.get("data", {}).get("kaggle_date_end", "2010-02-01")

# Taille des chunks pour ne pas saturer la RAM (13M lignes au total)
CHUNK_SIZE = 500_000


def _clean_amount(series: pd.Series) -> pd.Series:
    """Retire le '$' et convertit en float. Gère les valeurs manquantes."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": None, "nan": None})
        .astype(float)
    )


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes Kaggle vers le schéma du pipeline."""
    return df.rename(
        columns={
            "id": "transaction_id",
            "date": "timestamp",
            "client_id": "account_id",
            "use_chip": "transaction_type",
        }
    )


def build_transactions() -> pd.DataFrame:
    """Lit transactions_data.csv par chunks, filtre la période, nettoie."""
    src = RAW_KAGGLE / "transactions_data.csv"
    if not src.exists():
        raise FileNotFoundError(f"Fichier introuvable : {src}")

    log.info("Lecture de %s par chunks de %d lignes", src, CHUNK_SIZE)

    kept: list[pd.DataFrame] = []
    total_read = 0

    for chunk in pd.read_csv(src, chunksize=CHUNK_SIZE):
        total_read += len(chunk)
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        mask = (chunk["date"] >= DATE_START) & (chunk["date"] < DATE_END)
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            kept.append(filtered)
        log.info(
            "chunk lu=%d | retenus=%d | cumul=%d",
            len(chunk), len(filtered), sum(len(k) for k in kept),
        )

    if not kept:
        raise ValueError(
            f"Aucune transaction entre {DATE_START} et {DATE_END}. "
            "Vérifie la période ou le fichier source."
        )

    df = pd.concat(kept, ignore_index=True)
    log.info("Total retenu avant nettoyage : %d lignes (sur %d lues)", len(df), total_read)

    # Nettoyage
    df = _rename_columns(df)

    df["amount"] = _clean_amount(df["amount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Réordonne les colonnes pour un CSV lisible
    ordered = [
        "transaction_id", "timestamp", "account_id", "card_id", "amount",
        "transaction_type", "merchant_id", "merchant_city", "merchant_state",
        "zip", "mcc", "errors",
    ]
    df = df[ordered]

    # Stats rapides pour le log
    log.info(
        "Période couverte : %s → %s",
        df["timestamp"].min(), df["timestamp"].max(),
    )
    log.info("Montant : min=%.2f max=%.2f moyenne=%.2f",
             df["amount"].min(), df["amount"].max(), df["amount"].mean())
    log.info("Types de transaction : %s", df["transaction_type"].value_counts().to_dict())

    return df


def copy_reference_files() -> None:
    """Copie cards_data.csv et users_data.csv tels quels dans data/raw/."""
    for name in ("cards_data.csv", "users_data.csv"):
        src = RAW_KAGGLE / name
        if not src.exists():
            log.warning("Fichier de référence manquant : %s (ignoré)", src)
            continue
        dst = RAW_OUT / name.replace("_data", "")
        shutil.copy2(src, dst)
        log.info("Copié : %s → %s", src, dst)


def main() -> None:
    RAW_OUT.mkdir(parents=True, exist_ok=True)

    log.info("=== Adaptation du dataset Kaggle ===")
    df = build_transactions()

    out_tx = RAW_OUT / "transactions.csv"
    df.to_csv(out_tx, index=False)
    log.info("Écrit : %s (%d lignes, %.1f Mo)",
             out_tx, len(df), out_tx.stat().st_size / 1e6)

    copy_reference_files()
    log.info("=== Terminé ===")


if __name__ == "__main__":
    main()