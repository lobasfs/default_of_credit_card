import pandas as pd
import numpy as np
from typing import Tuple


def create_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаем агрегированные признаки.

    Args:
        df: Входные данные.
        
    Returns:
        DataFrame с новыми признаками.
    """
    df = df.copy()

    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['avg_payment_status'] = df[pay_cols].mean(axis=1)
    df['max_payment_status'] = df[pay_cols].max(axis=1)
    df['min_payment_status'] = df[pay_cols].min(axis=1)

    df['late_payment_count'] = (df[pay_cols] > 0).sum(axis=1)

    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['max_bill_amt'] = df[bill_cols].max(axis=1)
    df['bill_amt_trend'] = df['BILL_AMT1'] - df['BILL_AMT6']

    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
    df['max_pay_amt'] = df[pay_amt_cols].max(axis=1)
    df['pay_amt_trend'] = df['PAY_AMT1'] - df['PAY_AMT6']

    df['payment_ratio'] = df['avg_pay_amt'] / (df['avg_bill_amt'] + 1)

    df['credit_utilization'] = df['avg_bill_amt'] / (df['LIMIT_BAL'] + 1)
    
    return df


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков из демографических данных
    
    Args:
        df: Входные данные.
        
    Returns:
        DataFrame с новыми признаками.
    """
    df = df.copy()
    
    # Age binning
    df['age_group'] = pd.cut(
        df['AGE'], 
        bins=[0, 25, 35, 45, 55, 100], 
        labels=['18-25', '26-35', '36-45', '46-55', '56+']
    )
    
    # Credit limit binning
    df['credit_limit_group'] = pd.qcut(
        df['LIMIT_BAL'], 
        q=5, 
        labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        duplicates='drop'
    )
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняем шаги создания новых признаков
    
    Args:
        df: Входные данные.

    Returns:
        DataFrame с новыми признаками.
    """
    df = create_payment_features(df)
    df = create_demographic_features(df)
    
    return df


def get_feature_columns() -> Tuple[list, list]:
    numerical_features = [
        'LIMIT_BAL', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        # Engineered features
        'avg_payment_status', 'max_payment_status', 'min_payment_status',
        'late_payment_count', 'avg_bill_amt', 'max_bill_amt', 'bill_amt_trend',
        'avg_pay_amt', 'max_pay_amt', 'pay_amt_trend', 'payment_ratio', 'credit_utilization'
    ]
    
    categorical_features = [
        'SEX', 'EDUCATION', 'MARRIAGE',
        'age_group', 'credit_limit_group'
    ]
    
    return numerical_features, categorical_features


if __name__ == "__main__":
    # Example usage
    from src.data.load_data import load_data
    
    df = load_data()
    df_engineered = engineer_features(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Engineered shape: {df_engineered.shape}")
    print(f"\nNew columns: {[col for col in df_engineered.columns if col not in df.columns]}")
