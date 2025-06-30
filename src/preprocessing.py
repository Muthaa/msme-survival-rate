import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df: pd.DataFrame):
    """
    Clean and prepare data for training.
    """
    # Drop missing or irrelevant columns (example)
    df = df.dropna()
    
    # Feature selection (replace with real column names)
    features = ['capital', 'employees', 'owner_education', 'uses_digital_tools']
    target = 'survived'  # Binary target column: 1 or 0
    
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
