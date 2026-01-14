import os
import pandas as pd
from pathlib import Path


def download_data(data_dir: str = "data/raw"):
    """
    Скачиваем датасет с Kaggle
    
    Prerequisites:
    - Установить kaggle: pip install kaggle
    - Внести Kaggle API credentials в ~/.kaggle/kaggle.json
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    os.system(
        f"kaggle datasets download -d uciml/default-of-credit-card-clients-dataset "
        f"-p {data_dir} --unzip"
    )
    
    print(f"Data downloaded to {data_dir}")


def load_data(filepath: str = "data/raw/UCI_Credit_Card.csv") -> pd.DataFrame:
    """
    Загружаем датасет
    
    Args:
        filepath: путь к файоу

    """
    df = pd.read_csv(filepath)
    
    # Rename target column for clarity
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default'})
    
    return df


def get_feature_names():
    feature_info = {
        'LIMIT_BAL': 'Amount of given credit',
        'SEX': 'Gender (1=male, 2=female)',
        'EDUCATION': 'Education (1=graduate school, 2=university, 3=high school, 4=others)',
        'MARRIAGE': 'Marital status (1=married, 2=single, 3=others)',
        'AGE': 'Age in years',
        'PAY_0': 'Repayment status in September',
        'PAY_2': 'Repayment status in August',
        'PAY_3': 'Repayment status in July',
        'PAY_4': 'Repayment status in June',
        'PAY_5': 'Repayment status in May',
        'PAY_6': 'Repayment status in April',
        'BILL_AMT1': 'Bill statement amount in September',
        'BILL_AMT2': 'Bill statement amount in August',
        'BILL_AMT3': 'Bill statement amount in July',
        'BILL_AMT4': 'Bill statement amount in June',
        'BILL_AMT5': 'Bill statement amount in May',
        'BILL_AMT6': 'Bill statement amount in April',
        'PAY_AMT1': 'Previous payment amount in September',
        'PAY_AMT2': 'Previous payment amount in August',
        'PAY_AMT3': 'Previous payment amount in July',
        'PAY_AMT4': 'Previous payment amount in June',
        'PAY_AMT5': 'Previous payment amount in May',
        'PAY_AMT6': 'Previous payment amount in April',
        'default': 'Default payment (1=yes, 0=no)'
    }
    return feature_info


if __name__ == "__main__":
    # Скачиваем данные, если их нет
    if not os.path.exists("data/raw/UCI_Credit_Card.csv"):
        print("Downloading data...")
        download_data()
    
    # Загружаем данные
    df = load_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nTarget distribution:\n{df['default'].value_counts()}")
