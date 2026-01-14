import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn
import yaml


def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Создаем пайплайн боработки
    
    Args:
        numerical_features: список числовых признаков
        categorical_features: список категориальных признаков
        
    Returns:
        ColumnTransformer с шагами обработки
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])
    
    return preprocessor


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Высчитываем метрики классификации.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        y_pred_proba: Вероятности
        
    Returns:
        Словарь метрик
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }
    return metrics


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Стори ROC криву
    
    Args:
        y_true: Истинные метки
        y_pred_proba: Вероятности
        save_path: путь для сохранения графика
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ROC curve saved to {save_path}")
    
    return plt.gcf()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Сторим confusion matrix
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        save_path: путь для сохранения графика
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Confusion matrix saved to {save_path}")
    
    return plt.gcf()


def train_model(
    model_type: str = "logistic_regression",
    data_dir: str = "data/processed",
    mlflow_tracking_uri: str = "mlruns",
    experiment_name: str = "credit_card_default",
    tune_hyperparameters: bool = False,
    **model_params
):
    """
    Треинровка модели с MLflow tracking
    
    Args:
        model_type: Тип модели ('logistic_regression', 'random_forest', 'gradient_boosting')
        data_dir: директория с обработанными данными
        mlflow_tracking_uri: MLflow tracking URI
        experiment_name: название эксперимента
        tune_hyperparameters: надо ли тюнить гиперпараметры.
        **model_params: Дополнительные параметры модели.
    """
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("\n1. Loading data...")
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    
    X_train = train_df.drop('default', axis=1)
    y_train = train_df['default']
    X_test = test_df.drop('default', axis=1)
    y_test = test_df['default']
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    print("\n2. Preparing features...")
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"   Numerical features: {len(numerical_features)}")
    print(f"   Categorical features: {len(categorical_features)}")

    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    
    # Define model
    print(f"\n3. Creating model: {model_type}")
    
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42, **model_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42, **model_params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42, **model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    with mlflow.start_run(run_name=f"{model_type}"):

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_features", len(X_train.columns))
        
        for param, value in model_params.items():
            mlflow.log_param(param, value)

        if tune_hyperparameters:
            print("\n4. Tuning hyperparameters...")
            
            if model_type == "logistic_regression":
                param_grid = {
                    'classifier__C': [0.01, 0.1, 1, 10],
                    'classifier__penalty': ['l2'],
                    'classifier__solver': ['lbfgs', 'liblinear']
                }
            elif model_type == "random_forest":
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            elif model_type == "gradient_boosting":
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_
            
            print(f"   Best parameters: {grid_search.best_params_}")
            mlflow.log_params({f"best_{k}": v for k, v in grid_search.best_params_.items()})
        else:
            print("\n4. Training model...")
            pipeline.fit(X_train, y_train)
        
        print("   Training complete!")

        print("\n5. Evaluating model...")
        
        # Train set
        y_train_pred = pipeline.predict(X_train)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)

        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
            print(f"   Train {metric}: {value:.4f}")
        
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
            print(f"   Test {metric}: {value:.4f}")

        print("\n6. Creating plots...")
        Path("reports/figures").mkdir(parents=True, exist_ok=True)

        roc_fig = plot_roc_curve(y_test, y_test_proba, "reports/figures/roc_curve.png")
        mlflow.log_figure(roc_fig, "roc_curve.png")
        plt.close()

        cm_fig = plot_confusion_matrix(y_test, y_test_pred, "reports/figures/confusion_matrix.png")
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        plt.close()

        print("\n7. Logging model...")
        mlflow.sklearn.log_model(pipeline, "model")
        
        print("\n✓ Training complete!")
        print("=" * 60)
    
    return pipeline


if __name__ == "__main__":
    pipeline = train_model(
        model_type="logistic_regression",
        tune_hyperparameters=False,
        C=1.0
    )
