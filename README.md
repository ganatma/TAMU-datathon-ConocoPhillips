# Equipment Failure Prediction - TAMU Datathon 2023 (2nd Place Solution)

## Competition Overview
[NASA Turbofan Engine Failure Prediction](https://www.kaggle.com/competitions/equipfails/overview)
**Task:** Predict failure of turbofan engines using sensor data
**Challenge:** Highly imbalanced dataset (1.67% failure rate)
**Result:** 2nd Place on Private Leaderboard (F1-score: 0.8044)

## Solution Approach

### Data Preprocessing
1. **Missing Value Handling**: Removed columns with >65% missing values
2. **NA Imputation**: Filled remaining NAs with median values from training set
3. **Train-Test Split**: 70-30 stratified split for validation

### Modeling
- **Algorithm**: LightGBM Classifier
- **Class Imbalance Handling**: Instance weighting (6:1 for failure:normal)
- **Hyperparameters**:
  - Learning rate: 0.12
  - Number of leaves: 200
  - Training iterations: 100

## Repository Structure
```
train.py # Training script
predict.py # Inference script
requirements.txt # Dependency list
data/ # Input data directory
train.csv # Training data
test.csv # Test data
outputs/ # Prediction outputs
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Training:
```bash
python train.py --data_path data/equip_failures_training_set.csv
```

3. Prediction:
```bash
python predict.py \
  --model model.pkl \
  --test_data data/equip_failures_test_set.csv \
  --output outputs/predictions.csv
```