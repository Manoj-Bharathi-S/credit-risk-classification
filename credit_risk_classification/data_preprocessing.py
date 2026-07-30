import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

def load_credit_data(data_path: str) -> pd.DataFrame:
    """
    Load credit risk data from CSV.
    """
    df = pd.read_csv(data_path)
    return df

def preprocess_and_prepare_features(df: pd.DataFrame, model_dir: Path):
    """
    Preprocess data (cleaning, label encoding), save artifacts, 
    and select features and target variable.
    """
    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
        
    # Drop missing values in account records
    df = df.dropna().reset_index(drop=True)
        
    features = ['age', 'sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'credit_amount', 'duration']
    target = 'risk'
    
    df_model = df[features + [target]].copy()
    
    # Categorical column encoding
    cat_cols = ['sex', 'housing', 'saving_accounts', 'checking_account']
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        joblib.dump(le, model_dir / f'{col}_encoder.pickle')
    
    # Target encoding (1 = Good, 0 = Bad)
    le_target = LabelEncoder()
    df_model[target] = le_target.fit_transform(df_model[target])
    joblib.dump(le_target, model_dir / 'target_encoder.pickle')
    
    # Separate Features and Target
    X = df_model.drop(columns=[target])
    y = df_model[target]
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=1):
    """
    Split data into training and testing sets
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)