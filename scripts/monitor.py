import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List
import time


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Расчет PSI.
    PSI < 0.1: Незначительные изменения
    0.1 <= PSI < 0.2: Умеренные изменения
    PSI >= 0.2: Существенные
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)

    expected_props = np.where(expected_props == 0, 0.0001, expected_props)
    actual_props = np.where(actual_props == 0, 0.0001, actual_props)

    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    
    return psi


def calculate_feature_psi(
    train_data: pd.DataFrame,
    new_data: pd.DataFrame,
    features: List[str]
) -> Dict[str, float]:
    """
    Расчет PSI для множества признаков.
    
    Args:
        train_data: тренировочный датасет
        new_data: новый датасет
        features: список признаков для расчета.
        
    Returns:
        Словарь с PSI для каждого признака.
    """
    psi_values = {}
    
    for feature in features:
        if feature in train_data.columns and feature in new_data.columns:
            if train_data[feature].dtype == 'object' or train_data[feature].dtype.name == 'category':
                train_dist = train_data[feature].value_counts(normalize=True, sort=False)
                new_dist = new_data[feature].value_counts(normalize=True, sort=False)

                all_categories = train_dist.index.union(new_dist.index)
                train_dist = train_dist.reindex(all_categories, fill_value=0.0001)
                new_dist = new_dist.reindex(all_categories, fill_value=0.0001)
                
                psi = np.sum((new_dist - train_dist) * np.log(new_dist / train_dist))
            else:
                psi = calculate_psi(
                    train_data[feature].values,
                    new_data[feature].values
                )
            
            psi_values[feature] = psi
    
    return psi_values


def monitor_model_predictions(
    api_url: str = "http://localhost:8000",
    train_data_path: str = "data/processed/train.csv",
    test_data_path: str = "data/processed/test.csv",
    sample_size: int = 100
):
    """
    Мониторинг модели.
    
    Args:
        api_url: URL
        train_data_path: Путь к тренировочным данным
        test_data_path: Путь к тестовым данным
        sample_size: размер сэмпла
    """
    print("=" * 60)
    print("MODEL MONITORING - DATA DRIFT DETECTION")
    print("=" * 60)

    print("\n1. Loading data...")
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    new_data = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
    
    print(f"   Train data: {len(train_data)} samples")
    print(f"   New data: {len(new_data)} samples")

    base_features = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3'
    ]

    print("\n2. Calculating PSI for features...")
    feature_psi = calculate_feature_psi(train_data, new_data, base_features)
    
    print("\n   Feature PSI values:")
    for feature, psi in sorted(feature_psi.items(), key=lambda x: x[1], reverse=True):
        status = "HIGH" if psi >= 0.2 else "MODERATE" if psi >= 0.1 else "OK"
        print(f"   {feature:15s}: {psi:.4f} {status}")

    print("\n3. Sending prediction requests to API...")
    predictions = []
    probabilities = []
    
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code != 200:
            print("API is not healthy, skipping prediction requests")
            return

        for idx, row in new_data.head(10).iterrows():
            input_data = row.drop('default').to_dict()

            response = requests.post(
                f"{api_url}/predict",
                json=input_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions.append(result['prediction'])
                probabilities.append(result['probability'])
                print(f"   Sample {idx}: Prediction={result['prediction']}, "
                      f"Probability={result['probability']:.3f}, "
                      f"Risk={result['risk_level']}")
            else:
                print(f"   ✗ Request failed for sample {idx}: {response.status_code}")
            
            time.sleep(0.1)

        if predictions:
            print("\n4. Calculating prediction drift...")
            
            train_proba = train_data['default'].values
            new_proba = np.array(probabilities)
            
            pred_psi = calculate_psi(train_proba, new_proba)
            status = "⚠️ HIGH DRIFT" if pred_psi >= 0.2 else "⚠️ MODERATE DRIFT" if pred_psi >= 0.1 else "✓ NO DRIFT"
            
            print(f"   Prediction PSI: {pred_psi:.4f} {status}")
        
    except requests.exceptions.ConnectionError:
        print("\n   ⚠️ Could not connect to API. Make sure it's running:")
        print("      python -m src.api.app")
        print("      or")
        print("      docker-compose up")
    except Exception as e:
        print(f"\n   ✗ Error during monitoring: {e}")


    print("\n" + "=" * 60)
    print("MONITORING SUMMARY")
    print("=" * 60)
    
    high_drift_features = [f for f, psi in feature_psi.items() if psi >= 0.2]
    moderate_drift_features = [f for f, psi in feature_psi.items() if 0.1 <= psi < 0.2]
    
    if high_drift_features:
        print(f"\n HIGH DRIFT detected in {len(high_drift_features)} features:")
        for f in high_drift_features:
            print(f"   - {f}: PSI = {feature_psi[f]:.4f}")
        print("\n   Action: Consider retraining the model")
    elif moderate_drift_features:
        print(f"\n MODERATE DRIFT detected in {len(moderate_drift_features)} features:")
        for f in moderate_drift_features:
            print(f"   - {f}: PSI = {feature_psi[f]:.4f}")
        print("\n   Action: Monitor closely, may need retraining soon")
    else:
        print("\n No significant drift detected")
        print("   Model performance is likely stable")
    
    print("=" * 60)


if __name__ == "__main__":
    monitor_model_predictions()
