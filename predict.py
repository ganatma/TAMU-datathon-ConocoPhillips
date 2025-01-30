"""
Inference Script for Turbofan Failure Prediction
"""

import argparse
import pandas as pd
import numpy as np
import joblib

# Columns to drop (matches training)
DROP_COLS = ['sensor2_measure', 'sensor38_measure', 'sensor39_measure',
             'sensor40_measure', 'sensor41_measure', 'sensor42_measure',
             'sensor43_measure', 'sensor68_measure']

def predict(model, test_data, X_train):
    """Generate predictions with median imputation"""
    # Preprocess test data
    test_clean = test_data.drop(DROP_COLS, axis=1, errors='ignore')
    medians = X_train.median()
    test_processed = test_clean.fillna(medians).drop(['id'], axis=1)
    
    # Generate predictions
    preds = model.predict(test_processed)
    return np.round(preds).astype(int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Load artifacts
    model = joblib.load(args.model)
    test_df = pd.read_csv(args.test_data, na_values='na')
    X_train = pd.read_csv('equip_failures_training_set.csv', na_values='na').drop(DROP_COLS + ['id', 'target'], axis=1)
    
    # Generate predictions
    predictions = predict(model, test_df, X_train)
    
    # Save results
    output_df = pd.DataFrame({
        'id': test_df['id'],
        'target': predictions
    })
    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}") 