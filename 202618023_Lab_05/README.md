# Lab 05 --- Machine Learning with Scikit-learn and From Scratch

## Student Information

  -----------------------------------------------------------------------
  Field                               Details
  ----------------------------------- -----------------------------------
  **Student ID**                      202618023

  **Student Name**                    Drashti Akbari

  **Lab**                             Lab 05 --- Machine Learning with
                                      Scikit-learn and From Scratch

  **Dataset**                         UCI Productivity Prediction of
                                      Garment Employees
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 1. Objective

This lab implements and compares regression and classification workflows
using:

-   **Scikit-learn**
-   **Manual implementations using NumPy and Pandas**

The work is divided into three parts:

-   **Part A:** Machine Learning with Scikit-learn
-   **Part B:** Machine Learning From Scratch
-   **Part C:** Comparison and Optimization

The same train-test samples, target definitions, preprocessing strategy,
and evaluation metrics are used to make the comparison reproducible and
fair.

------------------------------------------------------------------------

## 2. Dataset

The dataset used is the **Garment Worker Productivity** dataset:

``` text
garments_worker_productivity.csv
```

The dataset contains production-related information including:

-   `date`
-   `quarter`
-   `department`
-   `day`
-   `team`
-   `targeted_productivity`
-   `smv`
-   `wip`
-   `over_time`
-   `incentive`
-   `idle_time`
-   `idle_men`
-   `no_of_style_change`
-   `no_of_workers`
-   `actual_productivity`

### Initial data analysis

The notebook performs:

-   Dataset shape and column inspection
-   Data-type inspection
-   Missing-value analysis
-   Duplicate-value checking
-   Descriptive statistics
-   Categorical-value analysis
-   Unique-value analysis
-   Numerical distributions
-   Boxplots
-   Actual-productivity distribution analysis
-   Correlation analysis

The `department` column is cleaned using whitespace stripping.

------------------------------------------------------------------------

# Part A --- Scikit-learn Implementation

## 3. Target Variables

### Regression Target

The regression target is:

``` text
actual_productivity
```

The task is to predict continuous productivity using:

``` python
LinearRegression()
```

### Classification Target

A binary classification target named `MeetsTarget` is created:

``` python
df["MeetsTarget"] = (
    df["actual_productivity"] >= df["targeted_productivity"]
).astype(int)
```

Therefore:

-   `1` → actual productivity meets or exceeds targeted productivity
-   `0` → actual productivity is below targeted productivity

For classification, `actual_productivity` is excluded from the input
features to prevent target leakage.

------------------------------------------------------------------------

## 4. Train-Test Split

A single fixed split is used for both regression and classification.

The notebook uses:

``` text
Test size      : 20%
Random state   : 42
```

Because the regression target is continuous, quintile bins are created
using `pd.qcut`. These regression bins are combined with the binary
classification target to construct the stratification variable.

The resulting `train_idx` and `test_idx` are reused throughout Part A
and Part B.

This ensures that the Scikit-learn and from-scratch implementations are
evaluated on exactly the same observations.

------------------------------------------------------------------------

## 5. Scikit-learn Preprocessing

Preprocessing is fitted using the training data and then applied to the
test data.

### Numerical features

``` text
Missing values
      ↓
Median imputation
      ↓
StandardScaler
```

### Categorical features

``` text
Missing values
      ↓
Most-frequent imputation
      ↓
One-hot encoding
```

The categorical encoder uses:

``` python
OneHotEncoder(handle_unknown="ignore")
```

This allows unseen test-set categories to be handled safely.

------------------------------------------------------------------------

## 6. Scikit-learn Linear Regression

Model:

``` python
LinearRegression()
```

Evaluation metrics:

-   MAE
-   RMSE
-   R²
-   Training time
-   Prediction time

### Result

  Metric                  Scikit-learn
  --------------------- --------------
  MAE                         0.103598
  RMSE                        0.143751
  R²                          0.259663
  Training Time (s)           0.008429
  Prediction Time (s)         0.000701

------------------------------------------------------------------------

## 7. Scikit-learn Logistic Regression

Model:

``` python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### Result

  Metric                  Scikit-learn
  --------------------- --------------
  Accuracy                    0.741667
  Precision                   0.765258
  Recall                      0.931429
  F1-score                    0.840206
  Training Time (s)           0.014317
  Prediction Time (s)         0.001030

------------------------------------------------------------------------

# Part B --- From-Scratch Implementation

## 8. Manual Preprocessing

The same `train_idx` and `test_idx` from Part A are reused.

### Numerical missing values

Training-set medians are calculated:

``` python
numeric_medians = X_train[numerical_cols].median()
```

These training-set medians are used to fill missing numerical values in
both training and test data.

### Categorical missing values

Training-set modes are calculated and used to fill missing categorical
values.

### One-hot encoding

Categorical variables are encoded using:

``` python
pd.get_dummies()
```

The test set is reindexed to exactly match the training-set encoded
columns.

### Feature scaling

Numerical features are standardized using statistics calculated from the
training set. The encoded categorical columns are not scaled.

------------------------------------------------------------------------

## 9. From-Scratch Linear Regression

Linear Regression is implemented using the closed-form least-squares
solution:

\[ `\beta `{=tex}= (X^TX)^{-1}X\^Ty \]

The implementation uses NumPy matrix operations rather than
Scikit-learn.

### Comparison

  Metric                  Scikit-learn   From Scratch
  --------------------- -------------- --------------
  MAE                         0.103598       0.103598
  RMSE                        0.143751       0.143751
  R²                          0.259663       0.259662
  Training Time (s)           0.008429       0.000960
  Prediction Time (s)         0.000701       0.000098

The predictive results are essentially identical, showing that the
manually implemented least-squares solution reproduces the Scikit-learn
result closely for this dataset.

------------------------------------------------------------------------

## 10. From-Scratch Logistic Regression

The Logistic Regression model is implemented manually using NumPy.

The implementation includes:

-   Sigmoid function
-   Probability prediction
-   Class prediction
-   Binary log-loss
-   Gradient calculation
-   Gradient descent
-   L2 regularization
-   Convergence checking
-   Manual calculation of classification metrics

The sigmoid function is:

\[ `\sigma`{=tex}(z)=`\frac{1}{1+e^{-z}}`{=tex} \]

The gradient-descent update is:

\[ `\beta`{=tex}\_{new} =
`\beta`{=tex}-`\alpha`{=tex}`\nabla `{=tex}J(`\beta`{=tex}) \]

where:

-   (`\alpha`{=tex}) = learning rate
-   (`\nabla `{=tex}J(`\beta`{=tex})) = gradient of the objective

------------------------------------------------------------------------

## 11. From-Scratch Logistic Regression Baseline

The baseline implementation uses gradient descent with the initial
configuration used in Part B.

### Result

  Metric                  From Scratch Baseline
  --------------------- -----------------------
  Accuracy                             0.733333
  Precision                            0.732218
  Recall                               1.000000
  F1-score                             0.845411
  Training Time (s)                    0.033294
  Prediction Time (s)                  0.000232

------------------------------------------------------------------------

# Part C --- Comparison and Optimization

## 12. Linear Regression Optimization

The from-scratch Linear Regression implementation already uses:

-   Vectorized NumPy operations
-   Closed-form least-squares computation

Its predictive performance is almost identical to Scikit-learn.

Therefore, no additional model-level optimization was necessary for
Linear Regression.

------------------------------------------------------------------------

## 13. Logistic Regression Optimization

Several optimization experiments were performed.

### Learning-rate tuning

Learning rates tested included:

``` text
0.001
0.005
0.01
0.05
0.10
```

The experiments showed that the learning rate affects:

-   Loss convergence
-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Training time

A learning rate of `0.01` was retained for the later L2 regularization
experiments.

------------------------------------------------------------------------

## 14. L2 Regularization

L2 regularization was added to the Logistic Regression objective:

\[ J(`\beta`{=tex}) = `\text{Logistic Loss}`{=tex} +
`\frac{\lambda}{2n}`{=tex} `\sum`{=tex}\_{j=1}\^{p}`\beta`{=tex}\_j\^2
\]

The intercept is not regularized.

Several values of (`\lambda`{=tex}) were tested.

A longer experiment showed that the unregularized model required
substantially more iterations to satisfy the strict convergence
tolerance, while larger L2 penalties converged earlier.

A focused local search was then performed over:

``` text
λ = 5
λ = 7.5
λ = 10
λ = 12.5
λ = 15
```

The `λ = 10` model converged after:

``` text
13,335 iterations
```

with:

``` text
Accuracy  = 0.737500
Precision = 0.745614
Recall    = 0.971429
F1-score  = 0.843672
```

------------------------------------------------------------------------

## 15. Convergence Comparison

Two converged models were explicitly compared.

### λ = 0

``` text
Learning rate      = 0.01
Maximum iterations = 100000
Iterations used    = 56221
Tolerance          = 1e-7
Converged          = True
```

Results:

  Metric                     Value
  --------------------- ----------
  Accuracy                0.745833
  Precision               0.768868
  Recall                  0.931429
  F1-score                0.842377
  Training Time (s)       4.348503
  Prediction Time (s)     0.000133

### λ = 10

``` text
Learning rate      = 0.01
Maximum iterations = 20000
Iterations used    = 13335
Tolerance          = 1e-7
Converged          = True
```

Results:

  Metric                     Value
  --------------------- ----------
  Accuracy                0.737500
  Precision               0.745614
  Recall                  0.971429
  F1-score                0.843672
  Training Time (s)       1.055112
  Prediction Time (s)     0.000184

The λ=10 model reached convergence with substantially fewer iterations
than the λ=0 model. It also produced a slightly higher F1-score than the
converged λ=0 model, while having lower accuracy and precision and
higher recall.

------------------------------------------------------------------------

## 16. Final Logistic Regression Configuration

The final selected from-scratch Logistic Regression configuration in the
notebook is:

``` text
Learning rate       : 0.01
L2 lambda           : 10
Maximum iterations  : 20000
Iterations used     : 13335
Tolerance           : 1e-7
Converged           : True
Classification threshold : 0.5
```

Final metrics:

``` text
Accuracy  : 0.737500
Precision : 0.745614
Recall    : 0.971429
F1-score  : 0.843672
```

------------------------------------------------------------------------

## 17. Final Classification Comparison

  -------------------------------------------------------------------------------------
  Implementation     Accuracy   Precision     Recall   F1-score   Training   Prediction
                                                                  Time (s)     Time (s)
  ---------------- ---------- ----------- ---------- ---------- ---------- ------------
  Scikit-learn       0.737500    0.743478   0.977143   0.844444   0.014317     0.001030

  From Scratch       0.733333    0.732218   1.000000   0.845411   0.033294     0.000232
  Baseline                                                                 

  From Scratch       0.737500    0.745614   0.971429   0.843672   1.027275     0.000263
  Optimized                                                                
  -------------------------------------------------------------------------------------

### Interpretation

The optimized from-scratch model closely matches the Scikit-learn
classification performance.

Compared with the baseline:

-   Accuracy increases from `0.733333` to `0.737500`.
-   Precision increases from `0.732218` to `0.745614`.
-   Recall decreases from `1.000000` to `0.971429`.
-   F1-score decreases slightly from `0.845411` to `0.843672`.

Thus, L2 regularization changes the precision-recall trade-off rather
than improving every metric simultaneously.

------------------------------------------------------------------------

## 18. Performance and Runtime Differences

### Predictive performance

Small differences between Scikit-learn and the from-scratch model are
expected because the implementations use different optimization
procedures.

The from-scratch Logistic Regression uses manually implemented gradient
descent, learning-rate settings, L2 regularization, convergence
tolerance, and a probability threshold of `0.5`.

These choices determine the learned coefficients and decision boundary.

Therefore, small differences in Accuracy, Precision, Recall, and
F1-score can occur even when the same train-test data and features are
used.

### Training runtime

Scikit-learn is faster because its Logistic Regression implementation
uses optimized numerical routines and solver implementations.

The from-scratch implementation explicitly performs repeated
gradient-descent updates. The final optimized model required `13,335`
iterations to reach the strict convergence tolerance, which increases
training time.

The final observed training times were:

``` text
Scikit-learn        : 0.014317 s
From Scratch        : 1.027275 s
```

Beating Scikit-learn's execution time is not required; the runtime
difference is explained by the different implementations and
optimization strategies.

### Prediction runtime

Prediction requires substantially less computation than training. The
from-scratch implementation uses direct NumPy operations for the matrix
multiplication, sigmoid calculation, and thresholding.

For this relatively small dataset, its measured prediction time was
lower than Scikit-learn's in the experiment:

``` text
Scikit-learn        : 0.001030 s
From Scratch        : 0.000263 s
```

These timings are implementation- and dataset-dependent and should not
be interpreted as a general statement that the manual implementation is
always faster.

------------------------------------------------------------------------

# 19. Overall Conclusion

This lab demonstrated the complete machine-learning workflow using both
Scikit-learn and manual implementations.

### Regression

The from-scratch Linear Regression implementation achieved virtually the
same MAE, RMSE, and R² as Scikit-learn.

### Classification

The from-scratch Logistic Regression achieved performance close to
Scikit-learn. Learning-rate tuning, L2 regularization, convergence
analysis, and iteration-budget experiments were used to improve and
understand the manual implementation.

The final selected Logistic Regression configuration used:

``` text
Learning rate = 0.01
L2 lambda     = 10
Tolerance     = 1e-7
```

and converged after `13,335` iterations.

The experiments demonstrate that:

-   Correct preprocessing is important for fair comparison.
-   The same train-test split must be reused.
-   Convergence does not necessarily mean better test-set performance.
-   Regularization changes the precision-recall trade-off.
-   Optimized library implementations can be substantially faster than
    educational from-scratch implementations.
-   Runtime and predictive performance should be evaluated separately.

------------------------------------------------------------------------

## 20. Repository Structure

``` text
Lab-05/
│
├── README.md
├── 201618023_Lab_05(2).ipynb
└── garments_worker_productivity.csv
```

Make sure the dataset filename in the repository matches the filename
used in the notebook.

------------------------------------------------------------------------

## 21. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   Jupyter Notebook

### Part A

Scikit-learn is used for:

-   Train-test splitting
-   Missing-value imputation
-   Standard scaling
-   One-hot encoding
-   Linear Regression
-   Logistic Regression
-   Evaluation metrics

### Part B

NumPy and Pandas are used for the manual implementation of:

-   Missing-value handling
-   One-hot encoding
-   Feature scaling
-   Linear Regression
-   Logistic Regression
-   Sigmoid
-   Log-loss
-   Gradient descent
-   L2 regularization
-   Prediction
-   Evaluation metrics

------------------------------------------------------------------------

## 22. Reproducibility

Important experiment settings:

``` text
Test size              = 0.20
Random state           = 42
Classification threshold = 0.5

Final learning rate    = 0.01
Final L2 lambda        = 10
Final tolerance        = 1e-7
Maximum iterations     = 20000
```

The same train-test indices are reused across the Scikit-learn and
from-scratch implementations.
