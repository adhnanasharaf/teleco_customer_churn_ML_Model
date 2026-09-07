"""
Model Training, Benchmarking, and Hyperparameter Tuning Module
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

from src.data_loader import get_train_test_data
from src.preprocessing import build_preprocessor, get_feature_names_after_preprocessing
from src.evaluate import (
    evaluate_model,
    find_optimal_threshold,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_candidate_classifiers() -> Dict[str, Any]:
    """Returns candidate classification algorithms for benchmarking."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=0.5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
    }


def benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5
) -> List[Dict[str, Any]]:
    """
    Evaluates candidate models using Stratified K-Fold cross validation.
    """
    logger.info("Starting model benchmarking with %d-fold cross-validation...", cv_folds)
    classifiers = get_candidate_classifiers()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf)
        ])

        cv_res = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        res_dict = {
            "model_name": name,
            "accuracy": float(np.mean(cv_res["test_accuracy"])),
            "precision": float(np.mean(cv_res["test_precision"])),
            "recall": float(np.mean(cv_res["test_recall"])),
            "f1": float(np.mean(cv_res["test_f1"])),
            "roc_auc": float(np.mean(cv_res["test_roc_auc"])),
            "std_roc_auc": float(np.std(cv_res["test_roc_auc"])),
        }
        results.append(res_dict)
        logger.info(
            "Model: %-20s | ROC-AUC: %.4f | F1: %.4f | Recall: %.4f | Acc: %.4f",
            name, res_dict["roc_auc"], res_dict["f1"], res_dict["recall"], res_dict["accuracy"]
        )

    # Sort by ROC-AUC descending
    results = sorted(results, key=lambda x: x["roc_auc"], reverse=True)
    return results


def tune_best_model(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Performs hyperparameter grid search for the selected champion model.
    """
    logger.info("Tuning hyperparameters for best model: %s", best_model_name)
    preprocessor = build_preprocessor()

    if "Gradient Boosting" in best_model_name:
        base_clf = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 1.0],
        }
    elif "Random Forest" in best_model_name:
        base_clf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [6, 8, 10],
            "classifier__min_samples_split": [5, 10, 15],
        }
    elif "Logistic Regression" in best_model_name:
        base_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        param_grid = {
            "classifier__C": [0.05, 0.1, 0.5, 1.0, 5.0],
            "classifier__solver": ["lbfgs", "saga"],
        }
    else:
        base_clf = AdaBoostClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
        }

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_clf)
    ])

    grid_search = GridSearchCV(
        full_pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    logger.info("Best Grid Search ROC-AUC: %.4f with params: %s", grid_search.best_score_, grid_search.best_params_)
    return grid_search.best_estimator_


def extract_feature_importance(pipeline: Pipeline, sample_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Extracts feature names and corresponding importance scores/coefficients."""
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    feature_names = get_feature_names_after_preprocessing(preprocessor, sample_df)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    return feature_names, importances


def train_and_export_pipeline():
    """
    Complete orchestration:
    - Data loading
    - Benchmarking
    - Model selection & tuning
    - Threshold calibration
    - Evaluation & plot generation
    - Model export to models/
    """
    logger.info("Loading Telco Customer Churn dataset...")
    X_train, X_test, y_train, y_test = get_train_test_data()

    # Step 1: Benchmark models
    benchmark_results = benchmark_models(X_train, y_train)
    best_model_info = benchmark_results[0]
    logger.info("Champion Model identified: %s (CV ROC-AUC: %.4f)", best_model_info["model_name"], best_model_info["roc_auc"])

    # Step 2: Tune champion model
    best_pipeline = tune_best_model(best_model_info["model_name"], X_train, y_train)

    # Step 3: Threshold calibration
    train_proba = best_pipeline.predict_proba(X_train)[:, 1]
    optimal_thresh = find_optimal_threshold(y_train.values, train_proba, metric="f1")
    logger.info("Calibrated optimal decision threshold for F1: %.3f", optimal_thresh)

    # Step 4: Final Evaluation on Holdout Test Set
    metrics_test, y_proba_test, y_pred_test = evaluate_model(
        best_pipeline, X_test, y_test, threshold=optimal_thresh
    )
    logger.info(
        "Test Performance -> Acc: %.4f | Rec: %.4f | Prec: %.4f | F1: %.4f | ROC-AUC: %.4f",
        metrics_test["accuracy"],
        metrics_test["recall"],
        metrics_test["precision"],
        metrics_test["f1"],
        metrics_test["roc_auc"]
    )

    # Step 5: Feature Importances
    feature_names, importances = extract_feature_importance(best_pipeline, X_train.head(10))

    # Top drivers list
    top_driver_indices = np.argsort(importances)[::-1][:15]
    top_drivers = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in top_driver_indices
    ]

    # Step 6: Generate and Save Visual Plots
    plot_roc_curve(y_test.values, y_proba_test, metrics_test["roc_auc"])
    plot_confusion_matrix(metrics_test["confusion_matrix"])
    plot_feature_importance(feature_names, importances, top_n=15)
    plot_model_comparison(benchmark_results)

    # Step 7: Export Artifacts
    model_path = MODELS_DIR / "best_churn_model.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info("Saved trained pipeline model to %s", model_path)

    metadata = {
        "model_name": best_model_info["model_name"],
        "optimal_threshold": optimal_thresh,
        "test_metrics": metrics_test,
        "top_drivers": top_drivers,
        "benchmark_summary": benchmark_results,
    }

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", meta_path)

    return metadata


if __name__ == "__main__":
    train_and_export_pipeline()
