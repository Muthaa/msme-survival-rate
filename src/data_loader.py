import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load MSME dataset from CSV.
    """
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
