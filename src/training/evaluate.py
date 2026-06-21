import pandas as pd
from datasets import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import os


MODEL_PATH = "Walid692/finsentinel-camembert"
TEST_CSV = "data/datasets/test.csv"
OUTPUT_FILE = "data/evaluation_results.json"
MAX_LENGTH = 128


def load_data(path: str) -> Dataset:
    """
    Load CSV and convert to HuggingFace Dataset.
    """
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df)


def evaluate():
    """
    Evaluate fine-tuned CamemBERT on test dataset.
    Loads model from HuggingFace Hub.
    Saves results to JSON.
    """
    print("Loading model from HuggingFace Hub...")
    print(f"  Model : {MODEL_PATH}")

    tokenizer = CamembertTokenizer.from_pretrained(MODEL_PATH)
    model = CamembertForSequenceClassification.from_pretrained(MODEL_PATH)

    print("\nLoading test dataset...")
    test_dataset = load_data(TEST_CSV)
    print(f"  Test samples : {len(test_dataset)}")

    print("\nTokenizing...")
    test_dataset = test_dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True
    )
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer = Trainer(model=model)

    print("\nRunning evaluation...")
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(axis=-1)
    labels = predictions.label_ids

    accuracy = round(accuracy_score(labels, preds), 4)
    f1 = round(f1_score(labels, preds, average="weighted"), 4)
    report = classification_report(
        labels, preds,
        target_names=["bearish", "bullish", "neutral"],
        output_dict=True
    )

    print("\nTest Results :")
    print(f"  Accuracy : {accuracy}")
    print(f"  F1 Score : {f1}")
    print()
    print(classification_report(
        labels, preds,
        target_names=["bearish", "bullish", "neutral"]
    ))

    results = {
        "model": MODEL_PATH,
        "test_samples": len(test_dataset),
        "accuracy": accuracy,
        "f1_weighted": f1,
        "per_class": report
    }

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved : {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()