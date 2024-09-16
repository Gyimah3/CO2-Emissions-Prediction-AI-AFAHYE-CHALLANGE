# CO2-Emissions-Prediction-AI-AFAHYE-CHALLANGE

# CO2 Emissions Prediction - Code Documentation

## Project Overview
This document provides a detailed explanation of the code used to achieve second place in the CO2 emissions prediction competition. The objective was to create machine learning models using open-source CO2 emissions data from satellite observations to predict carbon emissions.

## Data Sources
- Predictors: Sentinel-5P satellite data (ESA)
- Target: Ground truth CO2 emissions from GRACED and EDGAR
- Time period: January 2019 to November 2022
- Locations: Approximately 800 locations from 20 areas in South Africa

## Code Structure

### 1. Data Loading and Preprocessing
- Libraries imported: pandas, numpy, sklearn, lightgbm, catboost, xgboost, tqdm, matplotlib, seaborn
- Data loaded from CSV files: 'Trai.csv' and 'Test.csv' (that is how i named it)
- Memory optimization function `convert_types()` applied to reduce dataframe memory usage

### 2. Feature Engineering
- Location feature created from latitude and longitude
- Month feature extracted from year and week number
- Trigonometric features created for cyclical nature of months
- Rolling statistics (mean, max, min, sum, std, skew) calculated for various windows (17, 39, 52 weeks)
- Location-based statistics computed

### 3. Model Training
#### LightGBM Model
- StratifiedKFold cross-validation with 12 splits based on the 'month' column
- LightGBM regressor with custom hyperparameters
- Log transformation applied to target variable
- Early stopping used to prevent overfitting

#### CatBoost Model
- Single CatBoost regressor trained on entire dataset
- Custom hyperparameters and early stopping applied

### 4. Ensemble
- Simple average ensemble of LightGBM and CatBoost predictions

## Model Performance
The final ensemble achieved second place in the competition leaderboard.

## Running the Code
**I ran this code using kaggle notebook
1. Ensure all required libraries are installed (see requirements at the start of the notebook)
2. Place 'Trai.csv' and 'Test.csv' in the kaggle input directory
3. Run the script to generate predictions

## Future Improvements
- Feature selection to reduce model complexity
- Hyperparameter tuning using techniques like Bayesian optimization
- Exploration of other ensemble techniques (e.g., stacking)
- Incorporation of domain knowledge for feature engineering

repo link.... https://github.com/Gyimah3/CO2-Emissions-Prediction-AI-AFAHYE-CHALLANGE

