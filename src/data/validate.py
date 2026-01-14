import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd


def get_input_schema() -> DataFrameSchema:
    """
    Определяем схему для валидации данных.
    """
    schema = DataFrameSchema(
        {
            "ID": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
            "LIMIT_BAL": Column(float, Check.in_range(0, 1000000), nullable=False),
            "SEX": Column(int, Check.isin([1, 2]), nullable=False),
            "EDUCATION": Column(int, Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False),
            "MARRIAGE": Column(int, Check.isin([0, 1, 2, 3]), nullable=False),
            "AGE": Column(int, Check.in_range(18, 100), nullable=False),
            "PAY_0": Column(int, Check.in_range(-2, 9), nullable=False),
            "PAY_2": Column(int, Check.in_range(-2, 9), nullable=False),
            "PAY_3": Column(int, Check.in_range(-2, 9), nullable=False),
            "PAY_4": Column(int, Check.in_range(-2, 9), nullable=False),
            "PAY_5": Column(int, Check.in_range(-2, 9), nullable=False),
            "PAY_6": Column(int, Check.in_range(-2, 9), nullable=False),
            "BILL_AMT1": Column(float, nullable=False),
            "BILL_AMT2": Column(float, nullable=False),
            "BILL_AMT3": Column(float, nullable=False),
            "BILL_AMT4": Column(float, nullable=False),
            "BILL_AMT5": Column(float, nullable=False),
            "BILL_AMT6": Column(float, nullable=False),
            "PAY_AMT1": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "PAY_AMT2": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "PAY_AMT3": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "PAY_AMT4": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "PAY_AMT5": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "PAY_AMT6": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "default": Column(int, Check.isin([0, 1]), nullable=False),
        },
        strict=False,
        coerce=True,
    )
    return schema


def validate_data(df: pd.DataFrame, schema: DataFrameSchema = None) -> pd.DataFrame:
    """
    Валидация данных по схеме.
    
    Args:
        df: DataFrame с данными
        schema: Pandera схема.
        
    Raises:
        pandera.errors.SchemaError: если валидация провалилась.
    """
    if schema is None:
        schema = get_input_schema()
    
    try:
        validated_df = schema.validate(df, lazy=True)
        print("✓ Data validation passed")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("✗ Data validation failed:")
        print(err.failure_cases)
        raise


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Проверка метрик качества данных
    
    Args:
        df: DataFrame для проверки
        
    Returns:
        Справочник с метиками.
    """
    quality_metrics = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'target_distribution': df['default'].value_counts().to_dict() if 'default' in df.columns else None,
    }

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outliers[col] = int(outlier_count)
    
    quality_metrics['outliers'] = outliers
    
    return quality_metrics


if __name__ == "__main__":
    from src.data.load_data import load_data

    df = load_data()
    
    print("Validating data...")
    validated_df = validate_data(df)
    
    print("\nData quality metrics:")
    quality = check_data_quality(validated_df)
    for key, value in quality.items():
        print(f"{key}: {value}")
