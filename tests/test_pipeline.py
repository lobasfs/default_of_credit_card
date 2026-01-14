import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.features.engineer import (
    create_payment_features,
    create_demographic_features,
    engineer_features,
    get_feature_columns
)
from src.data.validate import validate_data, check_data_quality, get_input_schema
from src.models.train import calculate_metrics, create_preprocessing_pipeline


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'ID': range(1, n_samples + 1),
        'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),
        'SEX': np.random.choice([1, 2], n_samples),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples),
        'MARRIAGE': np.random.choice([1, 2, 3], n_samples),
        'AGE': np.random.randint(21, 70, n_samples),
        'PAY_0': np.random.randint(-1, 3, n_samples),
        'PAY_2': np.random.randint(-1, 3, n_samples),
        'PAY_3': np.random.randint(-1, 3, n_samples),
        'PAY_4': np.random.randint(-1, 3, n_samples),
        'PAY_5': np.random.randint(-1, 3, n_samples),
        'PAY_6': np.random.randint(-1, 3, n_samples),
        'BILL_AMT1': np.random.randint(0, 100000, n_samples),
        'BILL_AMT2': np.random.randint(0, 100000, n_samples),
        'BILL_AMT3': np.random.randint(0, 100000, n_samples),
        'BILL_AMT4': np.random.randint(0, 100000, n_samples),
        'BILL_AMT5': np.random.randint(0, 100000, n_samples),
        'BILL_AMT6': np.random.randint(0, 100000, n_samples),
        'PAY_AMT1': np.random.randint(0, 50000, n_samples),
        'PAY_AMT2': np.random.randint(0, 50000, n_samples),
        'PAY_AMT3': np.random.randint(0, 50000, n_samples),
        'PAY_AMT4': np.random.randint(0, 50000, n_samples),
        'PAY_AMT5': np.random.randint(0, 50000, n_samples),
        'PAY_AMT6': np.random.randint(0, 50000, n_samples),
        'default': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


class TestFeatureEngineering:
    
    def test_create_payment_features(self, sample_data):
        result = create_payment_features(sample_data)

        expected_cols = [
            'avg_payment_status', 'max_payment_status', 'min_payment_status',
            'late_payment_count', 'avg_bill_amt', 'max_bill_amt', 'bill_amt_trend',
            'avg_pay_amt', 'max_pay_amt', 'pay_amt_trend', 'payment_ratio', 'credit_utilization'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        assert not result['avg_payment_status'].isna().any()
        assert not result['late_payment_count'].isna().any()
    
    def test_create_demographic_features(self, sample_data):
        result = create_demographic_features(sample_data)

        assert 'age_group' in result.columns
        assert 'credit_limit_group' in result.columns

        assert result['age_group'].notna().all()
    
    def test_engineer_features(self, sample_data):
        original_shape = sample_data.shape
        result = engineer_features(sample_data)

        assert result.shape[1] > original_shape[1]

        assert result.shape[0] == original_shape[0]
    
    def test_get_feature_columns(self):
        numerical, categorical = get_feature_columns()
        
        assert isinstance(numerical, list)
        assert isinstance(categorical, list)
        assert len(numerical) > 0
        assert len(categorical) > 0


class TestDataValidation:
    
    def test_validate_data_success(self, sample_data):
        result = validate_data(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
    
    def test_validate_data_failure(self):
        invalid_data = pd.DataFrame({
            'ID': [1, 2],
            'LIMIT_BAL': [-1000, 2000000],
            'SEX': [1, 5],
            'AGE': [15, 25],
            'default': [0, 1]
        })

        with pytest.raises(Exception):
            validate_data(invalid_data)
    
    def test_check_data_quality(self, sample_data):
        metrics = check_data_quality(sample_data)
        
        assert 'total_rows' in metrics
        assert 'total_columns' in metrics
        assert 'missing_values' in metrics
        assert 'duplicate_rows' in metrics
        
        assert metrics['total_rows'] == len(sample_data)


class TestModelTraining:
    
    def test_calculate_metrics(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.7, 0.6])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'roc_auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_create_preprocessing_pipeline(self):
        numerical = ['feat1', 'feat2']
        categorical = ['cat1', 'cat2']
        
        pipeline = create_preprocessing_pipeline(numerical, categorical)
        
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')


class TestDataIntegrity:
    
    def test_no_missing_values_in_key_columns(self, sample_data):
        key_columns = ['ID', 'LIMIT_BAL', 'SEX', 'AGE', 'default']
        
        for col in key_columns:
            assert not sample_data[col].isna().any(), f"Missing values in {col}"
    
    def test_target_variable_binary(self, sample_data):
        assert set(sample_data['default'].unique()).issubset({0, 1})
    
    def test_age_range(self, sample_data):
        assert sample_data['AGE'].min() >= 18
        assert sample_data['AGE'].max() <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
