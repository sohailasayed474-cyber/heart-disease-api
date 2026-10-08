# Heart Disease Prediction API

End-to-end ML project: UCI Cleveland dataset -> EDA -> modeling -> tuning -> FastAPI.

## Problem
Predict whether a patient has heart disease from 13 clinical features (binary classification).

## Data
UCI Heart Disease (Cleveland), 303 rows. Missing values: `ca` (4) and `thal` (2).
Target converted to binary (0 = no disease, 1-4 = disease). No duplicate rows.

## Pipeline
- Imputation (median / most frequent), StandardScaler, OneHotEncoder
- Engineered features: age_group, chol_age_ratio, hr_reserve, high_bp
- Models compared (5-fold CV): Logistic Regression, Random Forest, Gradient Boosting, SVM, Voting Ensemble
- Tuning: GridSearchCV
- Best model: Gradient Boosting (learning_rate=0.1, max_depth=1, n_estimators=50)

## Results
- CV accuracy before tuning: LR 0.835, Voting 0.814, SVM 0.810, RF 0.802, GB 0.777
- Test accuracy: ~88% (61 patients), recall for disease ~86%
- Baseline (always "no disease"): ~54%

## API
- Built with FastAPI; input validated with Pydantic (type and range checks, invalid requests return 422).
- `POST /predict` takes the 13 original fields and builds the engineered features internally.

## Run locally
    pip install -r requirements.txt
    uvicorn app:app --reload
Open http://127.0.0.1:8000/docs

## Example request
    {"age":62,"sex":1,"cp":4,"trestbps":140,"chol":268,"fbs":0,"restecg":2,
     "thalach":160,"exang":0,"oldpeak":3.6,"slope":3,"ca":2,"thal":3}

## Limitations
Small dataset from one hospital and a small test set (61 rows). The deployed model was refit on all 303 rows after evaluation. Educational use only, not a medical device.
