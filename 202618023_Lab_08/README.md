# LAB 08: DECISION TREES AND SUPPORT VECTOR MACHINES

**Student Name:** Drashti Akbari  
**Student ID:** 202618023  
**Lab Number:** Lab_08

---

## Introduction

This lab explores two important supervised machine learning algorithms: Decision Trees and Support Vector Machines (SVMs). The work combines implementation, experimentation, visualization, and performance evaluation. Core Decision Tree operations and a linear SVM are implemented manually to understand how the algorithms work internally, and the results are compared with implementations provided by Scikit-learn. Kernel-based SVMs are also investigated to understand how nonlinear decision boundaries can classify data that cannot be separated well by a straight line.

The Banknote Authentication dataset is used for the main classification experiments. A synthetic nonlinear dataset is used to demonstrate how different SVM kernels behave. Across the lab, the emphasis is on understanding the algorithms, evaluating their performance fairly, and considering the trade-offs between model accuracy, complexity, interpretability, and computational cost.

## Objectives

- Understand how Decision Trees select features and thresholds for splitting data.
- Implement Gini impurity, entropy, and impurity-based split selection.
- Build a Decision Tree classifier from scratch and compare it with Scikit-learn.
- Understand the maximum-margin principle, hinge loss, and regularization in a linear SVM.
- Implement a linear SVM using NumPy and subgradient descent.
- Compare the manual linear SVM with Scikit-learn's linear SVM.
- Explore linear, polynomial, and Radial Basis Function (RBF) kernels.
- Visualize decision boundaries and study the effects of model parameters.
- Compare accuracy, precision, recall, F1-score, training time, and prediction time.
- Discuss interpretability, feature scaling, overfitting, and model complexity.

## Part A: Decision Trees

A Decision Tree is a supervised learning algorithm that makes predictions by repeatedly dividing data according to feature values. Internal nodes contain decision rules, branches represent the outcomes of those rules, and leaf nodes provide the predicted class. The algorithm builds a tree by finding splits that make the resulting groups more homogeneous with respect to the target class.

Two common criteria for measuring node impurity are Gini impurity and entropy.

Gini impurity:

`Gini = 1 - sum(p_i ** 2)`

Entropy:

`Entropy = -sum(p_i * log2(p_i))`

Here, `p_i` is the proportion of observations belonging to class `i` in a node. A node containing only one class has zero impurity. A node containing a mixture of classes has greater impurity.

For each candidate split, the impurity of the child nodes is combined according to their sizes. The best split is the one that reduces impurity most effectively under the selected criterion. When entropy is used, the reduction is described as information gain.

The manual implementation searches candidate features and thresholds, selects a split, and builds the tree recursively. Stopping conditions, including maximum depth and minimum sample requirements, limit how large the tree can grow. These conditions matter because a very deep tree can fit training data closely but may also learn noise and perform less effectively on unseen data.

The manual tree is compared with Scikit-learn's `DecisionTreeClassifier`. Different maximum-depth settings and splitting criteria are tested to observe how model complexity affects performance. Decision boundaries are also visualized using two selected features. These plots help show how a tree divides the feature space into regions assigned to different classes.

This part demonstrates the advantage of Decision Trees: their rules can often be inspected and explained. However, interpretability decreases as the tree becomes larger, and a deeper tree is not automatically a better model. Performance on held-out test data is important when selecting model settings.

## Part B: Linear Support Vector Machine from Scratch

A Support Vector Machine attempts to separate classes using a decision boundary. For a linear SVM, the boundary is a hyperplane described by:

`w · x + b = 0`

Here, `w` is the weight vector, `x` is the feature vector, and `b` is the bias. The SVM seeks a boundary that separates the classes while maximizing the margin around it.

A soft-margin SVM allows some observations to fall within the margin or be misclassified. The parameter `C` controls the trade-off between a wider margin and penalizing margin violations. A larger `C` places greater emphasis on reducing violations, whereas a smaller `C` applies stronger regularization and allows more violations when doing so helps maintain a simpler boundary.

The hinge loss for one observation is:

`max(0, 1 - y * (w · x + b))`

For the manual binary SVM, labels are represented as `-1` and `+1`. The regularized objective is:

`J(w, b) = 0.5 * ||w||^2 + C * mean(max(0, 1 - y * (Xw + b)))`

The first term penalizes large weights, while the second term penalizes observations that do not satisfy the margin requirement. The manual model uses batch subgradient descent to update the weights and bias over multiple iterations.

Feature scaling is applied because SVMs are sensitive to differences in feature magnitudes. The mean and standard deviation are calculated using the training set only. The same values are then used to transform the test set, preventing information from the test set from leaking into preprocessing.

The manual SVM is compared with Scikit-learn's `SVC` using a linear kernel. The separating hyperplane, margin lines, and margin-violating observations are visualized. Accuracy, precision, recall, and F1-score are used to assess classification performance.

The manual implementation may not match a library implementation automatically. Its results depend on the correctness of the gradient calculations, the learning rate, the number of iterations, the regularization setting, and whether optimization converges. These factors should be considered before interpreting differences between the two models.

## Part C: Kernel SVM and Nonlinear Decision Boundaries

Some datasets cannot be separated effectively by a straight line. In these situations, a nonlinear decision boundary may be needed. This part uses a synthetic dataset generated with Scikit-learn's `make_moons` function to demonstrate how SVM kernels handle nonlinear class shapes.

### Linear Kernel

The linear kernel creates a linear decision boundary. It is suitable when a straight line or hyperplane separates the classes reasonably well. It is relatively simple, but it may underfit when the class boundary is strongly nonlinear.

### Polynomial Kernel

The polynomial kernel can represent nonlinear relationships through polynomial combinations of features. Its behavior depends on parameters such as polynomial degree, `C`, and `coef0`. Increasing the degree can create a more flexible boundary, but excessive flexibility may lead to overfitting.

### RBF Kernel

The Radial Basis Function (RBF) kernel can model complex nonlinear boundaries. Its behavior depends strongly on `C` and `gamma`. The `gamma` parameter controls how localized the influence of individual training observations is. Lower values generally create smoother boundaries, while higher values can create more intricate boundaries around training examples.

The lab plots decision boundaries for the linear, polynomial, and RBF kernels and compares their behavior. Different polynomial degrees, regularization values, and RBF gamma values are tested to observe how hyperparameters affect model flexibility.

These experiments illustrate the balance between fitting the training data and generalizing to unseen observations. A highly flexible boundary may fit the training set very closely but perform worse on new data. Therefore, model selection should consider test performance as well as the complexity of the learned boundary.

## Part D: Final Comparison on the Banknote Authentication Dataset

The final part compares five models on the original Banknote Authentication dataset:

1. Manual Decision Tree
2. Scikit-learn Decision Tree
3. Manual Linear SVM
4. Scikit-learn Linear SVM
5. Scikit-learn RBF SVM

The dataset contains four numerical input features: `variance`, `skewness`, `curtosis`, and `entropy`. The target column is `class`, representing the binary class label.

A consistent stratified train-test split is used for all models so that each model is evaluated on the same held-out observations. Stratification helps preserve the approximate class proportions in both subsets. Decision Trees are trained on the original feature values because they generally do not require standardization. SVMs use standardized features, with the scaling parameters learned from the training set only and then applied to the test set. For the manual SVM, class labels are mapped to `-1/+1` for training and converted back to the original label format for evaluation as needed.

The models are compared using the following measures:

- **Accuracy:** The proportion of all predictions that are correct.
- **Precision:** Of the observations predicted as positive, the proportion that are truly positive.
- **Recall:** Of the actual positive observations, the proportion correctly identified.
- **F1-score:** The harmonic mean of precision and recall.
- **Training time:** The measured time required to fit a model.
- **Prediction time:** The measured time required to generate predictions for the test observations.

Accuracy provides an overall measure of correctness, while precision and recall describe different aspects of positive-class performance. F1-score combines precision and recall into one value. Training and prediction times indicate computational cost, although they may vary with the hardware, software environment, and implementation.

The comparison should not rely on one metric alone. A model with the highest test accuracy may be harder to explain or slower to train. A manual implementation may also run more slowly than an optimized library implementation even when both use a similar algorithm. The results should be used to discuss the practical trade-offs between predictive performance, speed, and interpretability.

## Interpretability, Scaling, Overfitting, and Complexity

Decision Trees are often easier to interpret because predictions can be explained as a sequence of feature-based rules. However, large trees can become difficult to understand and may overfit.

A linear SVM uses a linear boundary. Its learned weights can provide some insight into the contribution of features, but interpretation depends on feature scaling and the meaning of the input variables. An RBF SVM can represent more complex relationships, but its predictions are generally harder to explain using a short set of simple rules.

Feature scaling is important for SVMs because their optimization and distance calculations can be affected by different feature magnitudes. Decision Trees select thresholds on individual features and generally do not require scaling.

Overfitting can be investigated by comparing training and test performance. Very high training performance alongside substantially lower test performance may indicate that a model has learned patterns that do not generalize. Tree depth and SVM parameters such as `C`, polynomial degree, and `gamma` affect model complexity and should be chosen with generalization in mind.

## Libraries and Tools

The lab uses:

- **Python** for implementation and experiment execution.
- **NumPy** for numerical calculations and the manual SVM.
- **Pandas** for loading and processing tabular data.
- **Matplotlib** for feature plots, decision boundaries, and model comparisons.
- **Scikit-learn** for train-test splitting, standardization, reference models, synthetic data generation, and evaluation utilities.

## How to Run

1. Open `202618023_Lab_08(1).ipynb` in Jupyter Notebook or JupyterLab.
2. Install the required libraries.
3. Ensure the Banknote Authentication CSV file is in the expected location, or update the path in the notebook.
4. Run notebook cells in order because later cells may depend on functions and variables defined earlier.
5. Review the generated metrics, comparison tables, and visualizations.
6. Base the final discussion on the actual outputs from the notebook. Do not substitute example values for measured results.

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Conclusion

This lab provides practical experience with Decision Trees and Support Vector Machines, from implementing core algorithmic ideas to comparing models using a consistent evaluation process. The Decision Tree experiments demonstrate how impurity measures and split selection produce rule-based classifiers. The manual linear SVM demonstrates the role of hinge loss, regularization, feature scaling, and iterative optimization in learning a maximum-margin classifier.

The kernel experiments show how linear and nonlinear SVMs can create different decision boundaries and how hyperparameters influence model flexibility. The final Banknote Authentication comparison brings these concepts together by evaluating multiple models with a common train-test split and consistent metrics.

The final conclusions should be based on the measured results from the notebook. Considering classification metrics, runtime, interpretability, scaling requirements, and overfitting provides a more complete comparison than relying on accuracy alone.

---

## Student Details

**Student Name:** Drashti Akbari  
**Student ID:** 202618023  
**Lab:** Lab_08 — Decision Trees and Support Vector Machines
