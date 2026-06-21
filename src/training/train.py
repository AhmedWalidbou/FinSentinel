import os
import pandas as pd
import torch
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "camembert-base"
TRAIN_CSV = "data/datasets/train.csv"
VAL_CSV = "data/datasets/val.csv"
OUTPUT_DIR = "models/camembert_finsentinel"
MLFLOW_EXPERIMENT = "finsentinel_training"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5


def load_data(path: str) -> Dataset:
    """
    Load CSV and convert to HuggingFace Dataset.
    """
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df)


def tokenize_dataset(dataset: Dataset, tokenizer: CamembertTokenizer) -> Dataset:
    """
    Tokenize all texts in the dataset.
    """
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 score.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4)
    }


def train():
    """
    Full training pipeline.
    Load data, tokenize, train CamemBERT, log with MLflow.
    """
    print("Starting FinSentinel training...")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Max length : {MAX_LENGTH}")

    print("\nLoading tokenizer and model...")
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
    model = CamembertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    print("\nLoading datasets...")
    train_dataset = load_data(TRAIN_CSV)
    val_dataset = load_data(VAL_CSV)
    print(f"  Train : {len(train_dataset)} samples")
    print(f"  Val   : {len(val_dataset)} samples")

    print("\nTokenizing datasets...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="models/logs",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("max_length", MAX_LENGTH)
        mlflow.log_param("train_samples", len(train_dataset))

        print("\nTraining CamemBERT...")
        trainer.train()

        print("\nEvaluating...")
        results = trainer.evaluate()
        print(f"  Accuracy : {results['eval_accuracy']}")
        print(f"  F1 Score : {results['eval_f1']}")

        mlflow.log_metric("accuracy", results["eval_accuracy"])
        mlflow.log_metric("f1_score", results["eval_f1"])

        print(f"\nSaving model to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("\nTraining complete !")
        print(f"  Accuracy : {results['eval_accuracy']}")
        print(f"  F1 Score : {results['eval_f1']}")


if __name__ == "__main__":
    train()