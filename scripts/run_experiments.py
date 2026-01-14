import sys
sys.path.append('.')

from src.models.train import train_model


def run_experiments():
    """
    Запускаем эксперименты.
    """
    
    experiments = [
        {
            'name': 'Logistic Regression - Default',
            'model_type': 'logistic_regression',
            'tune_hyperparameters': False,
            'C': 1.0
        },
        {
            'name': 'Logistic Regression - Tuned',
            'model_type': 'logistic_regression',
            'tune_hyperparameters': True,
        },
        {
            'name': 'Random Forest - Default',
            'model_type': 'random_forest',
            'tune_hyperparameters': False,
            'n_estimators': 100,
            'max_depth': 10
        },
        {
            'name': 'Random Forest - Tuned',
            'model_type': 'random_forest',
            'tune_hyperparameters': True,
        },
        {
            'name': 'Gradient Boosting - Default',
            'model_type': 'gradient_boosting',
            'tune_hyperparameters': False,
            'n_estimators': 100,
            'learning_rate': 0.1
        },
        {
            'name': 'Gradient Boosting - Tuned',
            'model_type': 'gradient_boosting',
            'tune_hyperparameters': True,
        },
    ]
    
    print("=" * 80)
    print("RUNNING MULTIPLE EXPERIMENTS")
    print("=" * 80)
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'=' * 80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp['name']}")
        print(f"{'=' * 80}\n")

        params = {k: v for k, v in exp.items() if k != 'name'}
        
        try:
            train_model(**params)
            print(f"\n✓ Experiment {i} completed successfully!")
        except Exception as e:
            print(f"\n✗ Experiment {i} failed: {e}")
    
    print("\n\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print("\nView results in MLflow UI:")
    print("  mlflow ui --port 5000")
    print("  Then open http://localhost:5000")


if __name__ == "__main__":
    run_experiments()
