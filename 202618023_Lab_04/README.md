# Airbnb Price Prediction

deployed link: https://202618023drashtids605-4eauchq5syzarhu9g7mtet.streamlit.app/

## Project Overview

This project develops a machine learning system to predict the **nightly price of Airbnb listings in New York City** using the Airbnb NYC 2019 dataset.

The project covers:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Feature selection
- Train-test splitting
- Multiple regression models
- Hyperparameter tuning
- Model evaluation
- Model serialization
- Deployment using **Gradio** and **Streamlit**

---

## Dataset

The project uses the **AB_NYC_2019.csv** dataset.

- **Rows:** 48,895
- **Columns:** 16
- **Target variable:** `price`
- **Target meaning:** Airbnb nightly price in USD

### Original columns

```text
id
name
host_id
host_name
neighbourhood_group
neighbourhood
latitude
longitude
room_type
price
minimum_nights
number_of_reviews
last_review
reviews_per_month
calculated_host_listings_count
availability_365
```

---

## Important Data Patterns

### Price distribution

Airbnb prices are strongly right-skewed. Most listings are in the lower and middle price ranges, while a small number of listings have very high prices.

The dataset contains legitimate high-price listings, including:

- 239 listings above $1,000
- 86 listings above $2,000
- 20 listings above $5,000
- 3 listings at or above $10,000

These observations were **retained rather than automatically removed**, because expensive listings may represent legitimate luxury properties.

### Room type

Room type is an important predictor of price. In general:

- Entire home/apt listings have higher prices.
- Private rooms have lower prices.
- Shared rooms generally have the lowest prices.

### Location

Location is another major factor. Both `neighbourhood_group` and detailed `neighbourhood`, together with latitude and longitude, capture geographic differences in Airbnb prices.

### Limitations in the dataset

The dataset does not contain detailed property-level variables such as:

- Number of bedrooms
- Number of bathrooms
- Property size
- Amenities

The absence of these variables makes it difficult for the model to accurately predict rare luxury listings.

---

## Data Cleaning and Preprocessing

### 1. Removed non-predictive/unstructured columns

The following columns were removed:

```python
[
    'id',
    'name',
    'host_id',
    'host_name'
]
```

Reasons:

- `id` and `host_id` are identifiers.
- `name` is unstructured text and was not processed using NLP.
- `host_name` is not considered a reliable generalizable price predictor.

### 2. Missing values in `reviews_per_month`

Missing values were replaced with `0`:

```python
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
```

### 3. Feature engineering from `last_review`

`last_review` was converted to a date and transformed into:

```text
days_since_last_review
```

using **2019-12-31** as the reference date.

Listings without a review history were assigned:

```text
-1
```

The original `last_review` column was then removed.

### 4. Log transformation of target

Because `price` is highly skewed, the target was transformed using:

```python
data['log_price'] = np.log1p(data['price'])
```

Predictions were converted back to dollar prices using:

```python
np.expm1(prediction)
```

The original `price` was retained for evaluation and interpretation.

---

## Features Used

The final model uses 11 features:

```python
features = [
    'neighbourhood_group',
    'neighbourhood',
    'latitude',
    'longitude',
    'room_type',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365',
    'days_since_last_review'
]
```

### Categorical features

```python
categorical_features = [
    'neighbourhood_group',
    'neighbourhood',
    'room_type'
]
```

These were transformed using **One-Hot Encoding** with:

```python
handle_unknown='ignore'
```

### Numerical features

```python
numerical_features = [
    'latitude',
    'longitude',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365',
    'days_since_last_review'
]
```

Numerical variables were standardized using `StandardScaler`.

---

## Train-Test Split

The dataset was divided into:

- **80% training data**
- **20% test data**

A fixed random state was used:

```python
random_state=42
```

To ensure the train and test sets had a comparable price distribution, the continuous prices were divided into five quantile-based bins and used for stratification.

---

## Machine Learning Pipeline

The preprocessing and model were combined into a Scikit-learn `Pipeline`.

The general workflow is:

```text
Raw Airbnb Data
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Train/Test Split
       |
       v
Preprocessing
 ┌─────────────────────┐
 │ Numerical            │
 │ StandardScaler       │
 └─────────────────────┘
            +
 ┌─────────────────────┐
 │ Categorical          │
 │ OneHotEncoder        │
 └─────────────────────┘
       |
       v
XGBoost Regressor
       |
       v
Log-price prediction
       |
       v
Inverse transformation
np.expm1()
       |
       v
Predicted Airbnb Price ($)
```

---

## Models Evaluated

Several regression approaches were tested.

| Model | MAE ($) | RMSE ($) | R² |
|---|---:|---:|---:|
| Linear Regression | 69.67 | 203.64 | 0.136 |
| Random Forest | 64.62 | 212.29 | 0.061 |
| Log Linear Regression | 58.45 | 204.67 | 0.127 |
| Gradient Boosting | 64.82 | 199.08 | 0.174 |
| Tuned Gradient Boosting | 63.53 | 201.73 | 0.152 |
| Log Gradient Boosting | 55.85 | 201.76 | 0.152 |
| Weighted Log Gradient Boosting | 55.59 | 196.73 | 0.194 |
| XGBoost + log price | 54.24 | 199.12 | 0.174 |
| **Tuned XGBoost + log price** | **53.68** | **197.69** | **0.186** |

### Final selected model

The final deployment model is:

**Tuned XGBoost trained on `log_price`.**

Test-set performance:

- **MAE:** $53.68
- **RMSE:** $197.69
- **R²:** 0.186

The model was selected as the final deployment model based primarily on its strong dollar-scale MAE among the evaluated final candidates.

---

## XGBoost Hyperparameter Tuning

`RandomizedSearchCV` was used with:

- **30 parameter combinations**
- **3-fold cross-validation**
- `random_state=42`
- `n_jobs=-1`

The search considered:

```python
xgb_param_distributions = {
    'model__n_estimators': [200, 300, 400, 500, 600],
    'model__learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
    'model__max_depth': [3, 4, 5, 6, 7, 8],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__subsample': [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'model__gamma': [0, 0.1, 0.3, 0.5],
    'model__reg_alpha': [0, 0.01, 0.1, 1],
    'model__reg_lambda': [1, 2, 5, 10]
}
```

---

## Error by Price Range

The final tuned XGBoost model performs substantially better on common lower and middle price ranges than on rare expensive listings.

| Actual Price Range | MAE ($) |
|---|---:|
| $0–100 | 20.54 |
| $101–200 | 36.50 |
| $201–500 | 95.14 |
| $501–1,000 | 431.30 |
| $1,001–5,000 | 1,549.58 |
| $5,000+ | 6,479.57 |

This shows that the model's predictions become less reliable as prices become extremely high.

The model tends to **underpredict rare luxury listings**, largely because there are very few such observations and the available features do not describe detailed luxury/property characteristics.

---

## Model Saving

The complete trained pipeline was saved using `joblib`:

```python
joblib.dump(
    best_xgb,
    "airbnb_price_model.pkl"
)
```

Model metadata was saved separately:

```python
joblib.dump(
    model_info,
    "airbnb_model_info.pkl"
)
```

The saved pipeline includes the preprocessing steps and XGBoost model, so the deployment application does not need to manually perform encoding or scaling.

The saved model was also reloaded and tested. Predictions from the original and loaded model were identical:

```text
Predictions identical: True
```

---

## Project Structure

Recommended project structure:

```text
Airbnb_Price_Prediction/
│
├── AB_NYC_2019.csv
├── 202618023_Lab_04(2).ipynb
├── airbnb_price_model.pkl
├── airbnb_model_info.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

Create/activate your Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

### Requirements

```text
pandas
numpy
scikit-learn==1.9.1
xgboost
joblib
gradio
streamlit
```

---

## Running the Gradio Application

The project includes a Gradio interface for making predictions.

Run:

```bash
python app.py
```

Gradio will normally provide a local address similar to:

```text
http://127.0.0.1:7860
```

Open this address in a browser.

The application accepts listing characteristics such as:

- Neighbourhood group
- Neighbourhood
- Latitude
- Longitude
- Room type
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability
- Days since last review

and returns an estimated Airbnb nightly price.

---

## Streamlit Support

Streamlit is also included in `requirements.txt`, so the project can be deployed with a Streamlit application if required.

Run a Streamlit app with:

```bash
streamlit run app.py
```

If the application is written specifically as a Streamlit app, this provides a browser-based interface for prediction.

---

## Limitations

1. **Extreme price prediction**
   - Rare luxury listings are difficult to predict accurately.
   - The model substantially underpredicts some extreme prices.

2. **Limited property information**
   - Bedrooms, bathrooms, size, amenities, and other property-level attributes are not available in the selected features.

3. **Moderate overall R²**
   - The final R² of approximately 0.186 indicates that a large portion of price variation remains unexplained by the available variables.

4. **Historical dataset**
   - The dataset represents Airbnb listings from 2019, so the model should not be interpreted as a current NYC market pricing model without retraining on newer data.

---

## Conclusion

The project demonstrates an end-to-end Airbnb price prediction workflow, from exploratory analysis and preprocessing to model tuning and deployment.

The strongest final model was **Tuned XGBoost with log-transformed price**, achieving:

```text
MAE  = $53.68
RMSE = $197.69
R²   = 0.186
```

The analysis shows that **location and room type are important price-related factors**, while the skewed distribution and small number of luxury listings make high-price prediction challenging. Retaining legitimate high-priced listings and using a log-transformed target provides a more appropriate modeling strategy than simply deleting expensive observations.
