"""
NASA Turbofan Failure Prediction - Training Script
TAMU Datathon 2023 - 2nd Place Solution
"""

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

# Configuration
SEED = 222
DROP_COLS = ['sensor2_measure', 'sensor38_measure', 'sensor39_measure',
             'sensor40_measure', 'sensor41_measure', 'sensor42_measure',
             'sensor43_measure', 'sensor68_measure']

def load_data(file_path):
    """Load and preprocess training data"""
    df = pd.read_csv(file_path, na_values='na')
    
    # Remove problematic columns
    df = df.drop(DROP_COLS, axis=1, errors='ignore')
    
    # Separate features and target
    X = df.drop(['id', 'target'], axis=1)
    y = df['target']
    
    return X, y

def preprocess(X_train, X_test):
    """Handle missing values using median imputation"""
    medians = X_train.median()
    
    return (
        X_train.fillna(medians),
        X_test.fillna(medians)
    )

def train_model(X_train, y_train):
    """Train LightGBM classifier with class weights"""
    # Class weights for imbalance handling
    weights = np.where(y_train == 1, 6, 1)
    
    # LGBM Dataset format
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        weight=weights,
        free_raw_data=False
    )
    
    # Model parameters
    params = {
        'objective': 'binary',
        'learning_rate': 0.12,
        'num_leaves': 200,
        'verbosity': -1,
        'seed': SEED
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        keep_training_booster=True
    )
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--model_path', default='model.pkl')
    args = parser.parse_args()

    # Load and split data
    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    
    # Preprocess data
    X_train, X_test = preprocess(X_train, X_test)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Save model
    joblib.dump(model, args.model_path)
    print(f"Model saved to {args.model_path}") 