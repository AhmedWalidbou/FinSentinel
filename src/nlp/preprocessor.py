import pandas as pd
import re
import json
import os
from typing import List


INPUT_CSV = "data/processed/financial_sentiment.csv"
OUTPUT_CSV = "data/processed/financial_sentiment_clean.csv"
OUTPUT_TRAIN = "data/datasets/train.csv"
OUTPUT_VAL = "data/datasets/val.csv"
OUTPUT_TEST = "data/datasets/test.csv"


def clean_text(text: str) -> str:
    """
    Clean a single text sample.
    Removes HTML tags, extra spaces, URLs.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters but keep French accents
    text = re.sub(r'[^\w\s\.\,\!\?\-àâäéèêëîïôùûüçœæ]', ' ', text)

    # Remove extra spaces
    text = re.sub(r' +', ' ', text)

    # Strip
    text = text.strip()

    return text


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out low quality samples.
    Removes empty texts and very short texts.
    """
    initial = len(df)

    # Remove empty
    df = df[df["text"].notna()]
    df = df[df["text"].str.len() > 10]

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    removed = initial - len(df)
    print(f"  Filtered: {removed} samples removed")
    print(f"  Remaining: {len(df)} samples")
    return df


def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Split dataset into train / val / test.
    80% train, 10% val, 10% test.
    Stratified by label.
    """
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp["label"]
    )

    return train, val, test


def preprocess_dataset() -> None:
    """
    Full preprocessing pipeline.
    Load → Clean → Filter → Split → Save.
    """
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded: {len(df)} samples")

    print("\nCleaning texts...")
    df["text"] = df["text"].apply(clean_text)

    print("\nFiltering dataset...")
    df = filter_dataset(df)

    print("\nLabel distribution:")
    for label, count in df["label"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"  Label {label}: {count} ({pct:.1f}%)")

    os.makedirs("data/datasets", exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nClean dataset saved: {OUTPUT_CSV}")

    print("\nSplitting dataset (80/10/10)...")
    train, val, test = split_dataset(df)

    train.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8")
    val.to_csv(OUTPUT_VAL, index=False, encoding="utf-8")
    test.to_csv(OUTPUT_TEST, index=False, encoding="utf-8")

    print(f"  Train : {len(train)} samples → {OUTPUT_TRAIN}")
    print(f"  Val   : {len(val)} samples → {OUTPUT_VAL}")
    print(f"  Test  : {len(test)} samples → {OUTPUT_TEST}")

    print("\nPreprocessing done")


if __name__ == "__main__":
    preprocess_dataset()