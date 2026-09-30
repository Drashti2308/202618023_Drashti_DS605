# DS605 Lab 06 --- Image and Text Feature Extraction and Classification

## Overview

This project implements **DS605: Fundamentals of Machine Learning ---
Lab 06**, focused on feature extraction and traditional machine-learning
classification for:

-   **Part A:** Asphalt crack image classification
-   **Part B:** Email spam classification
-   **Part C:** Representation improvement experiments

The notebook follows the lab workflow:

**Data → Preprocessing → Feature Extraction / Representation →
Train-Test Split → Traditional ML Model → Evaluation → Representation
Improvement**

> **Note:** The current email dataset (`emails.csv`) is already a
> numerical word-frequency representation with 3,000 word columns.
> Therefore, the notebook does not apply CountVectorizer or TF-IDF to
> the current file. The text experiments are performed directly on the
> provided word-frequency features.

------------------------------------------------------------------------

## Project Structure

``` text
DS605_Lab6/
│
├── 202618023_Lab_06.ipynb
├── README.md
├── emails.csv
├── asphalt_image_features.csv
│
└── 448/
    ├── Cracks/
    └── NonCracks/
```

File names and folder structure can be adjusted according to the local
setup.

------------------------------------------------------------------------

# Part A --- Image Feature Extraction and Classification

## Dataset

**Asphalt Crack Dataset**

-   Total images: **400**
-   Crack images: **200**
-   Non-crack images: **200**
-   Image format used: `.jpg`
-   Common image size: **448 × 448**
-   Classes:
    -   `Cracks = 1`
    -   `NonCracks = 0`

The dataset is identified in the lab specification as the **Asphalt
Crack Dataset from Mendeley Data**.

## Image Preprocessing

Each image is:

1.  Loaded using OpenCV.
2.  Resized to **448 × 448**.
3.  Converted from BGR/RGB representation to grayscale.
4.  Used to calculate intensity and edge-based features.

Example grayscale representation:

``` text
Original image
      ↓
Resize to 448 × 448
      ↓
Grayscale image
      ↓
Numerical feature extraction
```

## Extracted Features

Six handcrafted image features were extracted:

  Feature                Description
  ---------------------- ----------------------------------------------
  `mean_brightness`      Average grayscale intensity
  `contrast`             Standard deviation of grayscale intensity
  `dark_pixel_ratio`     Proportion of pixels with intensity \< 50
  `bright_pixel_ratio`   Proportion of pixels with intensity \> 200
  `edge_count`           Number of detected Canny edge pixels
  `edge_density`         Edge count divided by total number of pixels

### Canny Edge Detection

Original Canny thresholds:

``` python
threshold1 = 100
threshold2 = 200
```

The extracted feature table contains **400 rows and 9 columns**,
including image name, six numerical features, label, and class name.

The feature table is saved as:

``` text
asphalt_image_features.csv
```

## Class-Wise Feature Observations

  Feature                    Cracks    NonCracks
  -------------------- ------------ ------------
  Mean Brightness          144.2441     157.5030
  Contrast                  35.3105      28.7567
  Dark Pixel Ratio           0.0143       0.0032
  Bright Pixel Ratio         0.0949       0.0739
  Edge Count             67,917.515   57,650.465
  Edge Density               0.3384       0.2872

These values show differences between the two image classes in
brightness, contrast, pixel intensity ratios, and edge characteristics.

## Train-Test Split

The image dataset was divided using:

-   **80% training**
-   **20% testing**
-   `random_state = 42`
-   Stratified split

Result:

-   Training: **320 images**
-   Testing: **80 images**
-   Training distribution: 160 Crack / 160 NonCrack
-   Testing distribution: 40 Crack / 40 NonCrack

StandardScaler was fitted only on the training data and then applied to
the test data.

## Models

Three traditional classifiers were evaluated:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest

### Part A Results

  ---------------------------------------------------------------------------------
  Model          Accuracy   Precision     Recall   F1 Score   Training   Prediction
                                                              Time (s)     Time (s)
  ------------ ---------- ----------- ---------- ---------- ---------- ------------
  Logistic         0.8500      0.8182     0.9000     0.8571     0.0106       0.0009
  Regression                                                           

  Decision         0.8625      0.8537     0.8750     0.8642     0.0058       0.0044
  Tree                                                                 

  Random           0.9500      0.9286     0.9750     0.9512     0.2942       0.0139
  Forest                                                               
  ---------------------------------------------------------------------------------

Confusion matrices were also generated for the classifiers.

------------------------------------------------------------------------

# Part B --- Email Spam Classification

## Dataset

**Email Spam Classification Dataset**

-   Total emails: **5,172**
-   Total columns: **3,002**
-   Email identifier column: `Email No.`
-   Word-frequency feature columns: **3,000**
-   Target column: `Prediction`

Target distribution:

  Class              Count   Percentage
  ---------------- ------- ------------
  Non-spam (`0`)     3,672       71.00%
  Spam (`1`)         1,500       29.00%

## Important Representation Note

The supplied `emails.csv` does **not contain raw email messages**.
Instead, it already contains numerical word-frequency columns.

Therefore, this notebook does **not** reconstruct raw email text and
does **not** apply CountVectorizer or TF-IDF to the current dataset.

The existing representation is used directly:

``` text
Email dataset
     ↓
3,000 word-frequency features
     ↓
Train-Test Split
     ↓
Traditional Classifier
     ↓
Evaluation
```

## Word Cloud

A word cloud was generated by summing the frequency of each word across
all emails.

The word cloud is used for visual exploration of the most frequent terms
in the dataset.

## Sparsity

The feature matrix contains:

-   Total feature values: **15,516,000**
-   Zero values: **14,641,889**
-   Sparsity: **94.37%**

This indicates that most word-frequency entries are zero.

## Train-Test Split

The data was split using:

-   **80% training**
-   **20% testing**
-   `random_state = 42`
-   Stratification on `Prediction`

Result:

-   Training: **4,137 emails**
-   Testing: **1,035 emails**
-   Training class distribution: 2,937 non-spam / 1,200 spam
-   Testing class distribution: 735 non-spam / 300 spam

## Models

Two classifiers were evaluated:

### 1. Multinomial Naive Bayes

Multinomial Naive Bayes is suitable for non-negative
word-frequency/count features.

Results:

-   Accuracy: **94.20%**
-   Precision: **86.81%**
-   Recall: **94.33%**
-   F1 Score: **90.42%**
-   Training Time: approximately **0.11 s**
-   Prediction Time: approximately **0.10 s**

Confusion Matrix:

``` text
[[692  43]
 [ 17 283]]
```

### 2. Logistic Regression

Results:

-   Accuracy: **98.26%**
-   Precision: **95.78%**
-   Recall: **98.33%**
-   F1 Score: **97.04%**
-   Training Time: approximately **9.87 s**
-   Prediction Time: approximately **0.09 s**

Confusion Matrix:

``` text
[[722  13]
 [  5 295]]
```

### Model Comparison

  Model                       Accuracy   Precision   Recall   F1 Score
  ------------------------- ---------- ----------- -------- ----------
  Multinomial Naive Bayes       0.9420      0.8681   0.9433     0.9042
  Logistic Regression           0.9826      0.9578   0.9833     0.9704

The experiment shows that Logistic Regression achieved higher predictive
performance, while Multinomial Naive Bayes required substantially less
training time.

------------------------------------------------------------------------

# Part C --- Improve the Representation

Part C was implemented for **both the text and image tasks**.

## Part C --- Text: Feature Reduction

The original email representation contained **3,000 word-frequency
features**.

To reduce computation, Chi-Square feature selection was used to retain
the **1,000 most informative features**.

``` text
Original representation
3,000 features
       ↓
Chi-Square feature selection
       ↓
1,000 features
       ↓
Logistic Regression
```

### Text Improvement Results

  Metric                  Original   Improved
  --------------------- ---------- ----------
  Features                   3,000      1,000
  Accuracy                  0.9826     0.9729
  Precision                 0.9578     0.9474
  Recall                    0.9833     0.9600
  F1 Score                  0.9704     0.9536
  Training Time (s)           9.87       4.73
  Prediction Time (s)       0.0887     0.0060

Feature dimensionality was reduced by **66.7%**.

The reduction substantially decreased computation time, but it also
caused a small decrease in predictive performance.

### Text Trade-off

The experiment demonstrates the trade-off between:

-   **Fewer features → lower computation time**
-   **More features → higher predictive performance**

For this dataset, reducing 3,000 features to 1,000 improved
computational efficiency but reduced accuracy and F1-score.

------------------------------------------------------------------------

## Part C --- Image: Canny Threshold Improvement

For the asphalt images, the original Canny thresholds were:

``` text
100 / 200
```

The improved experiment changed them to:

``` text
50 / 150
```

The purpose was to detect potentially finer crack edges.

### Image Improvement Results

  --------------------------------------------------------------------------
  Approach   Canny           Accuracy    Precision       Recall     F1 Score
             Threshold                                          
  ---------- ----------- ------------ ------------ ------------ ------------
  Original   100 / 200         0.8500       0.8182       0.9000       0.8571

  Improved   50 / 150          0.8500       0.8182       0.9000       0.8571
  --------------------------------------------------------------------------

The classification metrics remained unchanged.

The recorded training time was slightly higher for the improved
experiment, so the threshold change did not provide a measurable
classification benefit in this dataset.

### Image Trade-off

Lowering the Canny thresholds can detect additional edge structures, but
more detected edges do not necessarily provide more useful information
for classification.

In this experiment:

-   Classification performance: **unchanged**
-   Training time: **slightly increased**
-   Canny thresholds: **100/200 → 50/150**

Therefore, the original threshold configuration was sufficient for the
tested feature representation.

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Jupyter Notebook
-   OpenCV
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
-   WordCloud

### Machine Learning Techniques

-   Handcrafted image feature extraction
-   Canny edge detection
-   Standardization
-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Multinomial Naive Bayes
-   Chi-Square feature selection
-   Confusion Matrix
-   Accuracy
-   Precision
-   Recall
-   F1 Score

No CNN, deep-learning model, or pretrained image embedding was used.

------------------------------------------------------------------------

# How to Run

## 1. Install dependencies

``` bash
pip install opencv-python numpy pandas matplotlib scikit-learn wordcloud
```

## 2. Prepare the datasets

For Part A, place the image folders in the configured dataset directory:

``` text
448/
├── Cracks/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── NonCracks/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

For Part B, place:

``` text
emails.csv
```

in the notebook working directory.

## 3. Open the notebook

``` text
202618023_Lab_06.ipynb
```

## 4. Run the cells

Run the notebook from top to bottom so that:

1.  Image features are extracted.
2.  The image feature table is created.
3.  Image classifiers are trained and evaluated.
4.  Email data is inspected.
5.  The word cloud is generated.
6.  Email classifiers are trained and evaluated.
7.  Representation improvements are tested.
8.  Original and improved approaches are compared.

------------------------------------------------------------------------

# Key Observations

### Image Classification

The handcrafted image features provided useful information for
separating crack and non-crack images. Among the evaluated image
classifiers, the Random Forest experiment recorded **95.00% accuracy and
95.12% F1-score**.

### Email Classification

The existing word-frequency representation was highly effective for spam
classification. Logistic Regression achieved **98.26% accuracy and
97.04% F1-score**, while Multinomial Naive Bayes provided much faster
training.

### Representation Improvement

Reducing the text feature space from 3,000 to 1,000 features
substantially reduced computation time but caused a small performance
decrease.

Changing the Canny thresholds from 100/200 to 50/150 did not change the
image classification metrics in this experiment.

------------------------------------------------------------------------

# Deliverables

The repository should contain:

-   `202618023_Lab_06.ipynb`
-   `README.md`
-   `asphalt_image_features.csv`
-   Required datasets or dataset instructions
-   Relevant visualizations
-   Evaluation results
-   Confusion matrices
-   Part C comparison

The lab specification requests a **public GitHub repository link** as
the submission.
