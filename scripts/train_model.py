import sys
import yaml
import json
from pathlib import Path
import joblib

sys.path.append('.')
from src.models.train import train_model


def main():
    """Запуска пайплайна тренировки"""

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    train_params = params['train']

    print("Starting training...")
    pipeline = train_model(
        model_type=train_params['model_type'],
        tune_hyperparameters=train_params.get('tune_hyperparameters', False),
        **{k: v for k, v in train_params.items() 
           if k not in ['model_type', 'tune_hyperparameters']}
    )
    
    # Save model
    model_path = Path('models/model.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, model_path)
    print(f"\n✓ Model saved to {model_path}")

    metrics_path = Path('reports/metrics.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        'train_roc_auc': 0.0,
        'test_roc_auc': 0.0,
        'test_precision': 0.0,
        'test_recall': 0.0,
        'test_f1_score': 0.0
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
