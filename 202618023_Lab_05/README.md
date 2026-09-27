# Lab-05: Machine Learning with Scikit-learn and From Scratch

## Student Information

  -----------------------------------------------------------------------
  Field                               Details
  ----------------------------------- -----------------------------------
  **Student ID**                      202618023

  **Student Name**                    Drashti Akbari

  **Lab**                             Lab-05: Machine Learning with
                                      Scikit-learn and From Scratch

  **Dataset**                         Garment Worker Productivity Dataset
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 1. Objective

This lab implements and compares machine-learning models using two
approaches:

1.  **Scikit-learn implementation**
2.  **From-scratch implementation using NumPy and Pandas**

The work covers:

-   Data loading and exploratory data analysis
-   Data preprocessing
-   A regression problem using Linear Regression
-   A classification problem using Logistic Regression
-   Manual implementation of the same workflow
-   Comparison of predictive performance and execution time
-   Optimization of the from-scratch Logistic Regression implementation

------------------------------------------------------------------------

# 2. Dataset

The dataset used is:

``` text
garments_worker_productivity.csv
```

The dataset contains **1197 observations and 15 original columns**.

### Main variables

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

### Data quality observations

-   Dataset size: **1197 × 15**
-   Duplicate rows: **0**
-   Missing values occur in `wip`
-   Missing `wip` values: **506**
-   `actual_productivity` has 879 unique values
-   `actual_productivity` ranges from approximately **0.234 to 1.120**

The `department` column was cleaned using whitespace stripping before
modeling.

------------------------------------------------------------------------

# 3. Exploratory Data Analysis

The notebook performs:

-   Dataset shape and structure inspection
-   Data-type inspection
-   Missing-value analysis
-   Duplicate-value checking
-   Descriptive statistics
-   Categorical frequency analysis
-   Unique-value analysis
-   Numerical distributions
-   Histograms
-   Boxplots
-   Target distribution analysis
-   Correlation analysis

The mean of `actual_productivity` is approximately:

``` text
0.735091
```

and the standard deviation is approximately:

``` text
0.174488
```

The strongest positive numerical correlation with `actual_productivity`
in the EDA is `targeted_productivity`, with correlation approximately:

``` text
0.421594
```

------------------------------------------------------------------------

# Part A --- Scikit-learn Implementation

## 4. Classification Target

A binary target `MeetsTarget` is created from:

``` python
actual_productivity >= targeted_productivity
```

The resulting target is:

``` text
1 → Actual productivity meets or exceeds the target
0 → Actual productivity is below the target
```

For classification, `actual_productivity` is not used as an input
feature because it directly determines the classification target.

------------------------------------------------------------------------

## 5. Train-Test Split

A fixed train-test split is used so that the Scikit-learn and
from-scratch implementations are evaluated on the same observations.

The same train/test indices are reused throughout the comparisons.

------------------------------------------------------------------------

## 6. Preprocessing

The preprocessing workflow includes:

### Numerical features

``` text
Missing-value imputation
        ↓
Median imputation
        ↓
Standard scaling
```

### Categorical features

``` text
Missing-value imputation
        ↓
Most-frequent imputation
        ↓
One-hot encoding
```

The preprocessing parameters are learned from the training data and then
applied to the test data.

------------------------------------------------------------------------

# 7. Regression --- Scikit-learn Linear Regression

The regression target is:

``` text
actual_productivity
```

The model used is:

``` python
LinearRegression()
```

### Metrics

-   MAE
-   RMSE
-   R²
-   Training time
-   Prediction time

### Scikit-learn result

  Metric                    Result
  --------------------- ----------
  MAE                     0.103598
  RMSE                    0.143751
  R²                      0.259663
  Training Time (s)       0.006608
  Prediction Time (s)     0.000575

------------------------------------------------------------------------

# 8. Regression --- From Scratch

Linear Regression is implemented manually using NumPy and the
closed-form least-squares solution.

The implementation uses:

\[ `\beta `{=tex}= (X^TX)^{-1}X\^Ty \]

with vectorized NumPy matrix operations.

### Comparison

  Metric                  Scikit-learn   From Scratch
  --------------------- -------------- --------------
  MAE                         0.103598       0.103598
  RMSE                        0.143751       0.143751
  R²                          0.259663       0.259662
  Training Time (s)           0.006608       0.000944
  Prediction Time (s)         0.000575       0.000189

The predictive results are virtually identical, demonstrating that the
manual least-squares implementation reproduces the Scikit-learn
regression results closely.

Because the from-scratch implementation already uses vectorized NumPy
operations and a closed-form solution, no additional model-level
optimization was performed for Linear Regression.

------------------------------------------------------------------------

# Part B --- From-Scratch Logistic Regression

## 9. Logistic Regression Implementation

The Logistic Regression model was implemented manually using NumPy.

The implementation contains:

-   Parameter initialization
-   Sigmoid function
-   Probability prediction
-   Class prediction
-   Logistic loss
-   Gradient calculation
-   Gradient descent
-   Classification metrics
-   Convergence checking
-   L2 regularization

The sigmoid function is:

\[ `\sigma`{=tex}(z)=`\frac{1}{1+e^{-z}}`{=tex} \]

The class prediction uses a threshold of:

``` text
0.5
```

------------------------------------------------------------------------

# 10. Classification --- Scikit-learn vs From Scratch

### Initial comparison

  -------------------------------------------------------------------------------------
  Implementation     Accuracy   Precision     Recall   F1-Score   Training   Prediction
                                                                  Time (s)     Time (s)
  ---------------- ---------- ----------- ---------- ---------- ---------- ------------
  Scikit-learn       0.741667    0.765258   0.931429   0.840206   0.013553     0.001483

  From Scratch       0.733333    0.732218   1.000000   0.845411   0.039546     0.000201
  -------------------------------------------------------------------------------------

The baseline from-scratch model produced a different precision-recall
trade-off. It achieved recall of 1.0 but generated more false positives.

Its F1-score was slightly higher than the Scikit-learn model, while its
training time was longer because gradient descent performs repeated
parameter updates.

------------------------------------------------------------------------

# Part C --- Comparison and Optimization

## 11. Optimization Strategy

The optimization was focused on Logistic Regression because the Linear
Regression implementation already produced virtually identical
predictive results.

The optimization was performed in stages:

``` text
Baseline Logistic Regression
          ↓
Learning-rate experiment
          ↓
Learning-rate selection
          ↓
Convergence-tolerance experiment
          ↓
Tolerance selection
          ↓
L2 regularization experiment
          ↓
Final optimized model
```

------------------------------------------------------------------------

# 12. Learning-Rate Optimization

The learning rates tested were:

``` text
0.001
0.005
0.01
0.05
0.10
```

Initially, with a fixed 5000-iteration limit and tolerance of `1e-7`,
increasing the learning rate reduced the final training loss, but this
did not consistently improve test-set F1-score.

A second experiment evaluated convergence using tolerance `1e-5`.

### Converged learning-rate results

  ----------------------------------------------------------------------------------------------
    Learning   Iterations       Accuracy      Precision     Recall       F1-Score  Training Time
        Rate                                                                                 (s)
  ---------- ------------ -------------- -------------- ---------- -------------- --------------
       0.001         3242       0.733333       0.732218   1.000000       0.845411       0.212336

       0.005         1552       0.733333       0.732218   1.000000       0.845411       0.103198

       0.010         1702       0.737500       0.735294   1.000000   **0.847458**       0.101294

       0.050         1642       0.737500       0.754545   0.948571       0.840506       0.117484

       0.100         1338   **0.745833**   **0.761468**   0.948571       0.844784   **0.091915**
  ----------------------------------------------------------------------------------------------

The learning rate was subsequently set to:

``` text
0.1
```

for the tolerance and L2 experiments because it provided high accuracy
and precision with fewer iterations and lower observed training time.

------------------------------------------------------------------------

# 13. Tolerance Optimization

After selecting the learning rate, the convergence tolerance was
investigated.

The tested values were:

``` text
1e-4
5e-5
1e-5
```

### Results

  -------------------------------------------------------------------------------------------
    Tolerance   Iterations       Accuracy      Precision     Recall       F1-Score   Training
                                                                                     Time (s)
  ----------- ------------ -------------- -------------- ---------- -------------- ----------
         1e-4          172       0.733333       0.734177   0.994286       0.844660   0.018288

         5e-5          390       0.737500       0.745614   0.971429       0.843672   0.026318

     **1e-5**     **1338**   **0.745833**   **0.761468**   0.948571   **0.844784**   0.089972
  -------------------------------------------------------------------------------------------

A tolerance of:

``` text
1e-5
```

was selected because it produced the highest F1-score among the tested
tolerance values while achieving convergence.

A stricter tolerance would require additional iterations without
providing a meaningful improvement in predictive performance.

------------------------------------------------------------------------

# 14. L2 Regularization

L2 regularization was then investigated while keeping:

``` text
Learning rate = 0.1
Maximum iterations = 5000
Tolerance = 1e-5
```

The tested values were:

``` text
0
0.001
0.01
0.1
1
5
10
```

### Results

  --------------------------------------------------------------------------------------------------
      Lambda   Iterations       Accuracy      Precision         Recall       F1-Score  Training Time
                                                                                                 (s)
  ---------- ------------ -------------- -------------- -------------- -------------- --------------
           0         1338       0.745833       0.761468       0.948571       0.844784       0.104122

       0.001         1338       0.745833       0.761468       0.948571       0.844784       0.092568

        0.01         1336       0.745833       0.761468       0.948571       0.844784       0.086477

     **0.1**     **1311**   **0.745833**   **0.761468**   **0.948571**   **0.844784**   **0.082234**

           1         1126       0.737500       0.754545       0.948571       0.840506       0.081571

           5          764       0.745833       0.752212       0.971429   **0.847880**       0.054636

          10          583       0.737500       0.743478       0.977143       0.844444       0.039431
  --------------------------------------------------------------------------------------------------

### Selected L2 value

The final model uses:

``` text
L2 lambda = 0.1
```

The selection was based on preserving the same observed accuracy,
precision, recall, and F1-score as the unregularized model while
reducing iterations and observed training cost.

Although λ = 5 produced the highest F1-score in this experiment, λ = 0.1
preserved the complete baseline metric profile while providing a more
balanced optimization outcome.

------------------------------------------------------------------------

# 15. Final Optimized Model

The final from-scratch Logistic Regression configuration is:

``` text
Learning Rate          : 0.1
L2 Lambda              : 0.1
Maximum Iterations     : 5000
Iterations Used        : 1311
Tolerance              : 1e-5
Converged              : True
Classification Threshold : 0.5
```

### Final metrics

  Metric                  Final Optimized Model
  --------------------- -----------------------
  Accuracy                             0.745833
  Precision                            0.761468
  Recall                               0.948571
  F1-Score                             0.844784
  Training Time (s)                    0.097255
  Prediction Time (s)                  0.000112

------------------------------------------------------------------------

# 16. Final Classification Comparison

  ----------------------------------------------------------------------------------------------------------
  Implementation         Accuracy      Precision         Recall       F1-Score  Training Time     Prediction
                                                                                          (s)       Time (s)
  ---------------- -------------- -------------- -------------- -------------- -------------- --------------
  Scikit-learn           0.741667       0.765258       0.931429       0.840206       0.013553       0.001483

  From Scratch           0.733333       0.732218       1.000000       0.845411       0.039546       0.000201
  Baseline                                                                                    

  **From Scratch     **0.745833**   **0.761468**   **0.948571**   **0.844784**   **0.097255**   **0.000112**
  Optimized**                                                                                 
  ----------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 17. Baseline → Optimized Improvement

The final optimized model changes relative to the baseline as follows:

  Metric                  Baseline   Optimized          Change
  --------------------- ---------- ----------- ---------------
  Accuracy                0.733333    0.745833   **+0.012500**
  Precision               0.732218    0.761468   **+0.029250**
  Recall                  1.000000    0.948571   **-0.051429**
  F1-Score                0.845411    0.844784   **-0.000627**
  Training Time (s)       0.039546    0.097255   **+0.057709**
  Prediction Time (s)     0.000201    0.000112   **-0.000089**

The optimized model improves accuracy and precision, while recall
decreases. The F1-score changes only slightly.

The increase in training time is expected because the final model
includes iterative convergence and L2 regularization.

------------------------------------------------------------------------

# 18. Performance and Runtime Difference

The from-scratch Logistic Regression requires more training time than
Scikit-learn because it explicitly performs repeated gradient-descent
updates.

Scikit-learn uses optimized solver implementations and optimized
numerical routines, whereas the manual implementation executes the
optimization process explicitly.

The final optimized model also performs convergence checking and
regularization, which adds computational work.

Therefore:

``` text
Scikit-learn
→ optimized solver implementation
→ faster training
```

while:

``` text
From Scratch
→ explicit gradient descent
→ repeated parameter updates
→ convergence checking
→ L2 regularization
→ higher training time
```

The assignment does not require the from-scratch implementation to beat
Scikit-learn in execution time. The purpose of the optimization is to
improve or closely match predictive performance and reduce unnecessary
computation while explaining the remaining runtime difference.

------------------------------------------------------------------------

# 19. Key Findings

### Regression

The manually implemented Linear Regression produced almost identical
predictive performance to Scikit-learn:

-   MAE was identical to six decimal places.
-   RMSE was identical to six decimal places.
-   R² differed only in the sixth decimal place.

This confirms that the closed-form NumPy implementation correctly
reproduces the regression solution.

### Classification

The from-scratch Logistic Regression produced a slightly different
precision-recall trade-off from Scikit-learn.

The optimization experiments demonstrated that:

-   Lower training loss does not necessarily produce higher test-set
    F1-score.
-   Learning rate affects both convergence and classification
    performance.
-   A less strict convergence tolerance can reduce unnecessary
    iterations.
-   L2 regularization changes the model's coefficient constraints and
    can affect the precision-recall balance.
-   Scikit-learn remains faster because its optimization routines are
    highly optimized.

------------------------------------------------------------------------

# 20. Conclusion

This lab implemented regression and classification using both
Scikit-learn and manual NumPy/Pandas approaches.

The from-scratch Linear Regression reproduced the Scikit-learn
predictive results very closely.

For Logistic Regression, the manual implementation was progressively
optimized through:

1.  Learning-rate experimentation
2.  Convergence-tolerance tuning
3.  L2 regularization

The final selected configuration was:

``` text
Learning Rate = 0.1
L2 Lambda     = 0.1
Tolerance     = 1e-5
Maximum Iterations = 5000
```

The final model achieved:

``` text
Accuracy  = 0.745833
Precision = 0.761468
Recall    = 0.948571
F1-Score  = 0.844784
```

The results demonstrate the difference between optimizing a training
objective and optimizing test-set predictive performance, as well as the
computational difference between a manual iterative implementation and a
highly optimized machine-learning library.

------------------------------------------------------------------------

# 21. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   Jupyter Notebook

------------------------------------------------------------------------

# 22. Repository Structure

``` text
Lab-05/
│
├── README.md
├── 201618023_Lab_05(3).ipynb
└── garments_worker_productivity.csv
```

The notebook contains the complete Part A, Part B, and Part C
implementation and experiments.

------------------------------------------------------------------------

# 23. Reproducibility

Important final settings:

``` text
Classification threshold : 0.5
Final learning rate      : 0.1
Final L2 lambda          : 0.1
Final tolerance          : 1e-5
Maximum iterations       : 5000
```

The same train-test split is reused for the Scikit-learn and
from-scratch comparisons to maintain a consistent evaluation basis.
