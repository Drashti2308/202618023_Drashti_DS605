# DS605 Lab 07: Image Representation, PCA, Eigenfaces and Face Recognition

**Student ID:** 202618023  
**Student Name:** Drashti Akbari  
**Course:** DS605 — Fundamentals of Machine Learning  
**Lab:** 07

## Objective

Represent face images numerically, reduce their dimensionality using Principal Component Analysis (PCA), visualize eigenfaces, reconstruct sample faces, perform face recognition, and compare similarity-based recognition with traditional machine-learning classifiers.

## Dataset

- **Dataset:** AT&T Database of Faces (ORL)
- The dataset contains grayscale face images organized by subject.
- Each image is converted to grayscale and flattened into a numerical feature vector.
- Subject labels are retained for training and evaluation.

> Place the extracted dataset in the location expected by the notebook, or update the dataset path in the notebook to match your environment.

## Workflow

1. Load and inspect the face images.
2. Convert images to grayscale and flatten each image into a feature vector.
3. Create a reproducible train-test split with every subject represented in both sets.
4. Calculate the mean face and center the training data.
5. Implement PCA using covariance/eigendecomposition and Singular Value Decomposition (SVD).
6. Calculate cumulative explained variance and select components that preserve at least 90% variance.
7. Visualize eigenfaces.
8. Project face images into PCA space and reconstruct sample faces.
9. Compare reconstruction error and storage requirements.
10. Perform nearest-neighbor recognition using raw pixels and PCA representations.
11. Train and compare traditional classifiers using PCA features.
12. Inspect predictions, experiment with PCA component counts and classifier settings, and report results.

## Methods and Tools

- Python
- NumPy for numerical operations and PCA linear algebra
- Pillow for image loading
- Matplotlib for visualizations
- scikit-learn for traditional classifier comparisons and evaluation metrics
- Jupyter Notebook

**Implementation requirement:** PCA/eigenface calculations are implemented using NumPy linear-algebra operations rather than `scikit-learn`'s PCA implementation.

## Results

The following results were observed in the current notebook run. Results may vary if the dataset, split, preprocessing, or model settings change.

| Method | Features | Accuracy |
|---|---|---:|
| Nearest Neighbor | Raw pixels | 98.75% |
| Nearest Neighbor | PCA (99 components) | 98.75% |
| K-Nearest Neighbors (`k=3`) | PCA features | 92.50% |
| Logistic Regression | PCA features | 100.00% |

The raw-pixel and PCA nearest-neighbor methods each correctly classified 79 of the 80 test images. Logistic Regression correctly classified all 80 test images in this particular split. These results describe this test split and do not guarantee the same performance on other data.

## Repository Contents

Include the following in the public repository:

- `202618023_Lab_07.ipynb` — runnable notebook with implementation and results
- `README.md` — project description and observations
- Figures for cumulative explained variance, eigenfaces, and face reconstruction
- Model-comparison results and any additional evaluation outputs

## How to Run

1. Install the required packages:

   ```bash
   pip install numpy pillow matplotlib scikit-learn pandas jupyter
   ```

2. Download and extract the ORL face dataset.
3. Open `202618023_Lab_07.ipynb` in Jupyter Notebook or JupyterLab.
4. Update the dataset path in the notebook if necessary.
5. Run the notebook cells in order.

## Conclusion

This lab explores how PCA can represent face images using fewer features while retaining useful variation. Eigenfaces visualize the principal directions learned from the training images, and PCA representations can be used for similarity-based recognition and traditional classification. In the current experiment, nearest-neighbor recognition using 99 PCA components matched the raw-pixel baseline accuracy while using a much smaller feature representation.

## Submission

Submit the **public GitHub repository link** containing the notebook, figures, results, and this README file.
