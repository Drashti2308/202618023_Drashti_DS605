# Lab_03 Hotel Booking Cancellation Prediction

## 📌 Assignment Overview

This lab focuses on predicting whether a hotel booking will be **cancelled or not cancelled** using machine learning classification models.

The workflow includes:

- Dataset exploration
- Missing-value handling
- Data leakage prevention
- Outlier detection and removal
- Feature preprocessing
- Model training
- Model evaluation
- Comparison of different preprocessing techniques

---

## 👤 Student Information

| Field | Details |
|---|---|
| **Name** | Drashti Akbari |
| **Student ID** | 202618023 |
| **Lab** | Lab_03 |
| **Assignment Title** | Hotel Booking Cancellation Prediction |

---

## 📊 Dataset

**Dataset:** Hotel Booking Demand Dataset

**Dataset Link:**  
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

The dataset contains information about hotel reservations, including booking details, customer information, stay duration, room information, and other booking-related features.

### Target Variable

The target variable is:

```text
is_canceled
```

- `0` → Booking was **not cancelled**
- `1` → Booking was **cancelled**

---

## 🧹 Data Preprocessing

### Missing Values

Missing-value counts and percentages were calculated for every column.

The `company` column was removed because it contained a **very high percentage of missing values**.

### Data Leakage

The following columns were removed:

```text
reservation_status
reservation_status_date
```

These columns reveal information about the final outcome of a booking. Keeping them could cause **data leakage** and give the model information that would not be available at the time of prediction.

### Outlier Detection

Numerical features were checked using:

- Boxplots
- IQR method

Only **clear and extreme outliers** were removed so that valid but unusual hotel bookings were not unnecessarily deleted.

---

## 🔄 Train-Test Split

The same train-test split was used for all experiments:

```python
train_test_split(
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

This ensured that all four experiments were evaluated using the same data split and could therefore be compared fairly.

---

## ⚙️ Preprocessing Pipelines

### Pipeline A

For numerical features:

```text
KNNImputer(n_neighbors=5)
        ↓
StandardScaler
```

For categorical features:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

### Pipeline B

For numerical features:

```text
KNNImputer(n_neighbors=5)
        ↓
MinMaxScaler
```

For categorical features, the same preprocessing was used:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

`ColumnTransformer` and `Pipeline` were used so that preprocessing was **fitted only on the training data**, preventing data leakage.

---

## 🤖 Models

Two classification algorithms were tested:

### Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

### Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

A total of **four model-pipeline combinations** were evaluated:

1. Logistic Regression + Pipeline A
2. Logistic Regression + Pipeline B
3. Decision Tree + Pipeline A
4. Decision Tree + Pipeline B

---

## 📈 Model Performance

| Model + Pipeline | Training Accuracy | Testing Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + Pipeline A | 78.77% | 78.74% | 78.54% | 64.38% | 70.76% |
| Logistic Regression + Pipeline B | 78.73% | 78.67% | 78.54% | 64.11% | 70.59% |
| **Decision Tree + Pipeline A** | **99.71%** | **80.97%** | **75.72%** | **77.08%** | **76.40%** |
| Decision Tree + Pipeline B | 99.71% | 80.96% | 75.74% | 76.99% | 76.36% |

---

## 🔍 Final Observations

### 1. Best Overall Model

**Decision Tree + Pipeline A** produced the best overall result.

- Testing Accuracy: **80.97%**
- Precision: **75.72%**
- Recall: **77.08%**
- F1-Score: **76.40%**

It had the highest testing accuracy and F1-score among the four experiments.

### 2. Logistic Regression Comparison

Pipeline A performed slightly better than Pipeline B:

- Pipeline A F1-score: **70.76%**
- Pipeline B F1-score: **70.59%**

This shows that **StandardScaler performed slightly better than MinMaxScaler for Logistic Regression** in this experiment.

### 3. Effect of Scaling on Decision Tree

Scaling made almost no difference to the Decision Tree:

- Pipeline A F1-score: **76.40%**
- Pipeline B F1-score: **76.36%**

The difference is very small, which is expected because Decision Trees are generally not sensitive to feature scaling.

### 4. Possible Overfitting

The Decision Tree achieved **99.71% training accuracy** but only around **81% testing accuracy**.

The train-test accuracy gap was approximately **18.74 percentage points** for Pipeline A and **18.75 percentage points** for Pipeline B.

This large difference indicates **possible overfitting**.

### 5. Logistic Regression Generalization

Logistic Regression had almost the same training and testing accuracy:

- Pipeline A: **78.77% vs 78.74%**
- Pipeline B: **78.73% vs 78.67%**

The very small difference suggests **little evidence of overfitting**.

---

## 🎯 Confusion Matrix

Confusion matrices were created for the best models from each algorithm:

- **Best Logistic Regression:** Logistic Regression + Pipeline A
- **Best Decision Tree:** Decision Tree + Pipeline A

The confusion matrices helped compare how well each model identified:

- Correctly predicted non-cancelled bookings
- Incorrectly predicted cancelled bookings
- Incorrectly predicted non-cancelled bookings
- Correctly predicted cancelled bookings

The Decision Tree performed better at identifying cancelled bookings, which is reflected in its higher recall and F1-score.

---

## 🏆 Final Conclusion

Among the four experiments, **Decision Tree + Pipeline A (KNNImputer + StandardScaler)** gave the best overall performance with an **80.97% testing accuracy and 76.40% F1-score**.

StandardScaler had a small positive effect on Logistic Regression, while scaling had almost no effect on the Decision Tree. However, the Decision Tree showed a large gap between training and testing accuracy, indicating possible overfitting.

Overall, the experiment demonstrates how preprocessing choices can affect different machine learning models and why both **performance metrics and overfitting should be considered when selecting the final model**.

