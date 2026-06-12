from datasets import load_dataset
import json
import os
import pandas as pd


OUTPUT_FILE = "data/raw/financial_data.json"
OUTPUT_CSV = "data/processed/financial_sentiment.csv"


def load_financial_phrasebank() -> list:
    """
    Load Financial PhraseBank dataset from HuggingFace.
    """
    print("Loading Financial PhraseBank...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

    data = []
    for split in dataset:
        for item in dataset[split]:
            label = item["label"]
            label_text = ["bearish", "bullish", "neutral"][label]
            data.append({
                "text": item["text"],
                "label": label,
                "label_text": label_text,
                "source": "twitter_financial_news",
                "language": "en"
            })

    print(f"  → {len(data)} sentences loaded")
    return data


def load_fiqa_dataset() -> list:
    """
    Load FiQA sentiment dataset from HuggingFace.
    """
    print("Loading FiQA dataset...")
    try:
        dataset = load_dataset("pauri32/fiqa-2018")
        data = []
        for split in dataset:
            for item in dataset[split]:
                if "sentence" in item and "sentiment_score" in item:
                    score = item["sentiment_score"]
                    if score > 0.1:
                        label_text = "positive"
                        label = 2
                    elif score < -0.1:
                        label_text = "negative"
                        label = 0
                    else:
                        label_text = "neutral"
                        label = 1
                    data.append({
                        "text": item["sentence"],
                        "label": label,
                        "label_text": label_text,
                        "source": "fiqa",
                        "language": "en"
                    })
        print(f"  → {len(data)} sentences loaded")
        return data
    except Exception as e:
        print(f"  ERROR FiQA: {e}")
        return []


def save_dataset(data: list) -> None:
    """
    Save combined dataset to JSON and CSV.
    """
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved: {OUTPUT_FILE}")

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"CSV saved: {OUTPUT_CSV}")

    print(f"\nDataset statistics:")
    print(f"  Total samples : {len(df)}")
    print(f"  Label 0       : {len(df[df['label']==0])}")
    print(f"  Label 1       : {len(df[df['label']==1])}")
    print(f"  Label 2       : {len(df[df['label']==2])}")


if __name__ == "__main__":
    all_data = []

    phrasebank = load_financial_phrasebank()
    all_data.extend(phrasebank)

    fiqa = load_fiqa_dataset()
    all_data.extend(fiqa)

    save_dataset(all_data)
    print(f"\nTotal dataset: {len(all_data)} samples ready for training")