import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

from src.data.load_data import load_data
from src.data.validate import validate_data, check_data_quality
from src.features.engineer import engineer_features


def prepare_data(
    input_path: str = "data/raw/UCI_Credit_Card.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Подготовка данных: загрузка, валидация, фича инжениринг и сплит.
    
    Args:
        input_path: Путь до данных.
        output_dir: Директория, куда сохранять обработанные данные.
        test_size: Пропорция для сплита.
        random_state: рандом сид.
    """
    print("=" * 60)
    print("DATA PREPARATION PIPELINE")
    print("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n1. Loading data...")
    df = load_data(input_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    print("\n2. Validating data...")
    df = validate_data(df)

    print("\n3. Checking data quality...")
    quality_metrics = check_data_quality(df)
    print(f"   Missing values: {sum(quality_metrics['missing_values'].values())}")
    print(f"   Duplicate rows: {quality_metrics['duplicate_rows']}")
    print(f"   Target distribution: {quality_metrics['target_distribution']}")

    with open(f"{output_dir}/data_quality.yaml", 'w') as f:
        yaml.dump(quality_metrics, f)

    print("\n4. Engineering features...")
    df = engineer_features(df)
    print(f"   Total features: {len(df.columns)}")

    print("\n5. Splitting data...")

    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Separate features and target
    X = df.drop('default', axis=1)
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Train default rate: {y_train.mean():.2%}")
    print(f"   Test default rate: {y_test.mean():.2%}")

    print("\n6. Saving processed data...")
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    feature_info = {
        'features': X_train.columns.tolist(),
        'target': 'default',
        'n_features': len(X_train.columns),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
    }
    
    with open(f"{output_dir}/features.yaml", 'w') as f:
        yaml.dump(feature_info, f)
    
    print(f"\n✓ Data preparation complete!")
    print(f"   Files saved to {output_dir}/")
    print("=" * 60)
    
    return train_df, test_df


if __name__ == "__main__":
    prepare_data()
