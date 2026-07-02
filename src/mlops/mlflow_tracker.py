import mlflow
import mlflow.pytorch
from datetime import datetime
import json
import os


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "finsentinel"


def setup_mlflow():
    """
    Setup MLflow tracking.
    Uses local SQLite database.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking uri : {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment   : {EXPERIMENT_NAME}")


def log_training_run(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    train_samples: int,
    val_samples: int,
    accuracy: float,
    f1_score: float,
    epoch_results: list
) -> str:
    """
    Log a complete training run to MLflow.
    Returns the run ID.
    """
    setup_mlflow()

    with mlflow.start_run(run_name=f"camembert_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:

        mlflow.log_params({
            "model_name": model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "train_samples": train_samples,
            "val_samples": val_samples
        })

        mlflow.log_metrics({
            "final_accuracy": accuracy,
            "final_f1": f1_score
        })

        for i, epoch in enumerate(epoch_results):
            mlflow.log_metrics({
                f"epoch_{i+1}_accuracy": epoch.get("accuracy", 0),
                f"epoch_{i+1}_f1": epoch.get("f1", 0),
                f"epoch_{i+1}_loss": epoch.get("loss", 0)
            }, step=i+1)

        mlflow.set_tags({
            "model_type": "camembert",
            "task": "sentiment_analysis",
            "language": "fr",
            "domain": "finance",
            "huggingface_model": "Walid692/finsentinel-camembert"
        })

        run_id = run.info.run_id
        print(f"\nMLflow run logged : {run_id}")
        print(f"  Accuracy : {accuracy}")
        print(f"  F1 Score : {f1_score}")

        return run_id


def log_inference(text: str, label: str, score: float, latency_ms: float) -> None:
    """
    Log a single inference to MLflow.
    """
    setup_mlflow()

    with mlflow.start_run(run_name="inference"):
        mlflow.log_params({"text_length": len(text)})
        mlflow.log_metrics({
            "confidence_score": score,
            "latency_ms": latency_ms
        })
        mlflow.set_tag("predicted_label", label)


if __name__ == "__main__":
    run_id = log_training_run(
        model_name="camembert-base",
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        max_length=128,
        train_samples=10252,
        val_samples=1282,
        accuracy=0.7808,
        f1_score=0.7759,
        epoch_results=[
            {"accuracy": 0.7067, "f1": 0.641, "loss": 0.696},
            {"accuracy": 0.7683, "f1": 0.756, "loss": 0.609},
            {"accuracy": 0.7800, "f1": 0.772, "loss": 0.518}
        ]
    )
    print(f"\nRun ID : {run_id}")