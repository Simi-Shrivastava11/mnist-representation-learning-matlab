# MNIST Representation Learning in MATLAB

This project presents a comparative analysis of different representation methods for image classification on the **MNIST handwritten digits dataset** using **MATLAB**. The study compares **Raw Data**, **Principal Component Analysis (PCA)**, **Kernel PCA (kPCA)**, and **Locally Linear Embedding (LLE)** using **k-Nearest Neighbors (kNN)** as the classifier.

The goal of the project is to understand how dimensionality reduction affects classification performance, computational efficiency, and representation quality in a high-dimensional image classification setting.

## Overview

Image classification problems often involve high-dimensional input data, which can increase computational cost and make learning less efficient. In this project, I compare four ways of representing handwritten digit images before classification:

- **Raw Data** — original 784 pixel values
- **PCA** — linear dimensionality reduction
- **Kernel PCA** — nonlinear dimensionality reduction using an RBF kernel
- **LLE** — manifold learning based on local neighborhood structure

Each representation is evaluated with the same classifier, **kNN**, to compare methods under a consistent setup.

## Dataset

This project uses the **MNIST** dataset of handwritten digits.

Dataset details:
- **60,000 training images**
- **10,000 test images**
- each image is **28 × 28 grayscale**
- each image is flattened into a **784-dimensional feature vector**

The training set is further split into training and validation subsets for model selection.

## Methodology

### Data Preprocessing
The following preprocessing steps were used:

- loading image and label data from standard MNIST files
- normalizing pixel values to the range **[0, 1]**
- splitting the original training set into training and validation subsets
- flattening each image into a 784-dimensional vector

### Representation Methods

#### 1. Raw Data
The baseline method uses all 784 pixel values directly, without dimensionality reduction.

#### 2. PCA
PCA projects the data into a lower-dimensional linear subspace while preserving most of the variance. In this project, PCA retained **80% of the variance**, resulting in a **44-dimensional representation**.

#### 3. Kernel PCA
Kernel PCA extends PCA to nonlinear patterns using an **RBF kernel**. To handle memory constraints, the implementation uses a **Nyström approximation** with landmark sampling.

#### 4. LLE
LLE is a nonlinear manifold learning method that preserves local neighborhood relationships. Due to memory limitations, the embedding was learned on a subset of the training data.

### Classification
All methods were evaluated using **k-Nearest Neighbors (kNN)**.

- hyperparameter tuning was performed over `k = 1, 3, 5, 7, 9`
- **Euclidean distance** was used for Raw Data
- **Cosine distance** was used for PCA, kPCA, and LLE
- performance was measured on the held-out test set after model selection

## Results

The experiments showed that dimensionality reduction can preserve or improve classification performance while significantly reducing computational cost.

### Test Accuracy Summary
- **Raw Data:** 96.91%
- **PCA:** 97.23%
- **Kernel PCA:** 97.12%
- **LLE:** evaluated as part of the comparative analysis

Among all methods, **PCA achieved the best overall performance**, while also reducing dimensionality from 784 to 44 features.

### Key Takeaway
PCA provided the best balance of:
- classification accuracy
- computational efficiency
- reduced dimensionality

This suggests that a carefully chosen lower-dimensional representation can outperform the raw feature space in image classification tasks.

## Code Contents

The main MATLAB script includes:

- helper functions for reading MNIST image and label files
- evaluation functions for accuracy, precision, recall, F1 score, and confusion matrix
- cross-validation for kNN hyperparameter tuning
- implementations for:
  - Raw Data + kNN
  - PCA + kNN
  - Kernel PCA via Nyström + kNN
  - LLE + kNN
- comparative analysis plots and timing summaries

## Requirements

To run this project, you will need:

- **MATLAB**
- access to the **MNIST raw data files**
- required MATLAB functions/toolboxes used for dimensionality reduction utilities such as:
  - `compute_mapping`
  - `out_of_sample`

If these functions are not available in your MATLAB environment, you may need to install the corresponding dimensionality reduction toolbox.

## How to Run

1. Download the MNIST raw files and place them in the project directory:
   - `train-images-idx3-ubyte`
   - `train-labels-idx1-ubyte`
   - `t10k-images-idx3-ubyte`
   - `t10k-labels-idx1-ubyte`

2. Open MATLAB, navigate to the project folder, and run:

```matlab
main
```

3. The script will:
  - load and preprocess MNIST
  - train and evaluate kNN on Raw Data, PCA, Kernel PCA, and LLE
  - generate plots and confusion matrices
  - print accuracy, precision, recall, F1 score, and computation time for each method
  - save intermediate result files if save_intermediate = true is enabled.
  
**Notes**
  - PCA gave the best reported performance in the project, with about 97.23% accuracy while reducing dimensionality from 784 to 44 features.
  - Kernel PCA and LLE are more computationally expensive and may require substantial memory.
  - If compute_mapping or out_of_sample are missing, parts of the script will not run until those dependencies are added.

## Repository Structure

```text
.
├── README.md
├── main.m
├── report/
   └── Project.pdf
```

## What I Learned

This project strengthened my understanding of:

- dimensionality reduction for high-dimensional image data
- manifold learning and nonlinear embeddings
- model comparison under a controlled classification setup
- tradeoffs between accuracy and computational cost
- implementing end-to-end machine learning workflows in MATLAB

## Future Improvements

Possible next steps include:

- adding runtime and memory benchmarking tables
- comparing results with additional classifiers such as SVM
- reproducing the same analysis in Python
- improving documentation for external toolbox dependencies
- adding saved figures directly to the repository

## Report

The repository also includes the final project report with methodology, experimental setup, plots, and discussion of results.
