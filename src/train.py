"""
Training script for Titanic survival prediction model.

This script loads preprocessed data, builds a baseline model, evaluates it
with a validation split, and saves the trained model artifact.

Notes:
- Kaggle test set has no labels (Survived), so we evaluate using a validation split
  from the original training portion.
- We one-hot encode categorical columns to avoid sklearn fit errors.
- MLflow is used for experiment tracking, logging parameters, metrics, and model artifacts.
"""

import os
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.utils import load_data_safely


PREPROCESSED_PATH = "data/derived/titanic_merged_preprocessed.csv"
RAW_TRAIN_PATH = "data/raw/titanic_train.csv"


def load_preprocessed_data(preprocessed_path: str = PREPROCESSED_PATH) -> pd.DataFrame | None:
    """Load the preprocessed merged dataset."""
    if not os.path.exists(preprocessed_path):
        print(f"Error: Preprocessed data not found at {preprocessed_path}")
        print("Please run preprocessing first.")
        return None

    df = load_data_safely(preprocessed_path)
    print(f"Loaded preprocessed data: {df.shape}")
    return df


def get_original_train_size(raw_train_path: str = RAW_TRAIN_PATH, default_size: int = 891) -> int:
    """Get the original Kaggle train size to split merged data back into train/test parts."""
    if os.path.exists(raw_train_path):
        train_original = load_data_safely(raw_train_path)
        return len(train_original)
    return default_size


def split_merged_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split merged preprocessed dataframe into original train part and Kaggle test part
    based on raw train size.
    """
    train_size = get_original_train_size()
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    return train_df, test_df


def build_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build feature matrices for training and Kaggle test.

    - Train labels come from 'Survived'
    - Categorical features are one-hot encoded
    - Kaggle test columns are aligned to training columns
    """
    if "Survived" not in train_df.columns:
        raise ValueError("Column 'Survived' is missing in the training portion of the merged data.")

    # Separate X/y for training portion
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)

    # Kaggle test portion may or may not contain 'Survived' (shouldn't), drop if exists
    X_kaggle = test_df.drop(columns=["Survived"], errors="ignore")

    # One-hot encode to ensure numeric features only
    X = pd.get_dummies(X, drop_first=True)
    X_kaggle = pd.get_dummies(X_kaggle, drop_first=True)

    # Align Kaggle test columns to training columns
    X_kaggle = X_kaggle.reindex(columns=X.columns, fill_value=0)

    # Safety: ensure no NaNs
    X = X.fillna(0)
    X_kaggle = X_kaggle.fillna(0)

    return X, y, X_kaggle


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """
    Train a baseline Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility
        class_weight: Class weight strategy

    Returns:
        Trained RandomForestClassifier model
    """
    print("\n--- Training Baseline Model ---")
    print("Model: Random Forest Classifier")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )

    model.fit(X_train, y_train)
    print("Model training completed!")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """
    Evaluate the model and print key metrics.

    Returns:
        Dictionary containing evaluation metrics
    """
    print("\n--- Model Evaluation ---")

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average="weighted")
    val_recall = recall_score(y_val, y_val_pred, average="weighted")
    val_f1 = f1_score(y_val, y_val_pred, average="weighted")

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")

    print("\n--- Classification Report (Validation) ---")
    print(classification_report(y_val, y_val_pred))

    print("\n--- Confusion Matrix (Validation) ---")
    print(confusion_matrix(y_val, y_val_pred))

    # Feature importance (top 10)
    print("\n--- Top 10 Feature Importances ---")
    importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))

    # Return metrics dictionary for MLflow logging
    metrics = {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
    }

    return metrics


def save_model(
    model: RandomForestClassifier, output_path: str = "models/baseline_model.pkl"
) -> None:
    """Save the trained model artifact."""
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to {output_path}")


def save_feature_columns(
    columns: list[str], output_path: str = "models/feature_columns.json"
) -> None:
    """Save feature column order used during training for serving alignment."""
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(columns, f)
    print(f"✓ Feature columns saved to {output_path}")


def save_kaggle_predictions(
    model: RandomForestClassifier,
    X_kaggle: pd.DataFrame,
    raw_test_path: str = "data/raw/titanic_test.csv",
    output_path: str = "data/output/kaggle_predictions.csv",
) -> None:
    """
    Save Kaggle-style predictions file:
    PassengerId,Survived

    We read PassengerId from the raw Kaggle test file if available.
    """
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    preds = model.predict(X_kaggle).astype(int)

    if os.path.exists(raw_test_path):
        raw_test = load_data_safely(raw_test_path)
        if "PassengerId" in raw_test.columns and len(raw_test) == len(preds):
            sub = pd.DataFrame(
                {"PassengerId": raw_test["PassengerId"].astype(int), "Survived": preds}
            )
        else:
            # Fallback: generate an index-based id if PassengerId is missing/mismatched
            sub = pd.DataFrame({"PassengerId": np.arange(1, len(preds) + 1), "Survived": preds})
    else:
        sub = pd.DataFrame({"PassengerId": np.arange(1, len(preds) + 1), "Survived": preds})

    sub.to_csv(output_path, index=False)
    print(f"✓ Kaggle predictions saved to {output_path}")


def main(
    experiment_name: str = "titanic_survival_prediction",
    run_name: str = None,
    n_estimators: int = 200,
    max_depth: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Main training pipeline with MLflow tracking.

    Args:
        experiment_name: MLflow experiment name
        run_name: Name for this specific run (auto-generated if None)
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum depth of trees
        test_size: Validation split size
        random_state: Random seed for reproducibility
    """
    print("=" * 60)
    print("Titanic Survival Prediction - Baseline Model Training")
    print("=" * 60)

    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"baseline_rf_{n_estimators}trees_{max_depth}depth_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "test_size": test_size,
                "random_state": random_state,
                "class_weight": "balanced",
                "model_type": "RandomForestClassifier",
            }
        )

        df = load_preprocessed_data()
        if df is None:
            return

        # Log dataset info
        mlflow.log_params(
            {
                "train_samples": len(df),
                "n_features": len(df.columns) - 1,  # Excluding target
            }
        )

        # Split merged df back into train/test portions
        train_df, test_df = split_merged_df(df)

        # Build encoded features
        X, y, X_kaggle = build_features(train_df, test_df)

        # Validation split (evaluate properly without Kaggle test labels)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Kaggle test: {X_kaggle.shape}")

        # Log data split info
        mlflow.log_params(
            {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "n_features_encoded": X_train.shape[1],
            }
        )

        # Train
        model = train_baseline_model(
            X_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        # Evaluate
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model", registered_model_name="TitanicSurvivalPredictor")

        # Save model locally as well
        save_model(model)
        save_feature_columns(X_train.columns.tolist())

        # Optional: generate Kaggle predictions file
        save_kaggle_predictions(model, X_kaggle)

        # Log feature importance as artifact
        importance_df = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        importance_path = "data/output/feature_importance.csv"
        Path(os.path.dirname(importance_path)).mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        print("\n" + "=" * 60)
        print("Training pipeline completed successfully!")
        print(f"MLflow run: {run_name}")
        print(f"Experiment: {experiment_name}")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Titanic survival prediction model with MLflow tracking"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="titanic_survival_prediction",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this MLflow run (auto-generated if not provided)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="Number of trees in Random Forest"
    )
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth of trees")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state,
    )
