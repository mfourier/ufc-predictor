# 🔹 K-Nearest Neighbors (KNN)

## 📌 Overview  

**K-Nearest Neighbors (KNN)** is a **non-parametric**, **instance-based** machine learning algorithm commonly used for both **classification** and **regression** tasks. Predictions are made based on the similarity of input samples to nearby training instances, making minimal assumptions about the underlying data distribution.

---

## 🔍 How KNN Works  

The KNN algorithm follows three main steps:

1. **Compute Distances**: Calculate distances between the new input sample and every instance in the training dataset, typically using metrics such as **Euclidean**, **Manhattan**, or **Minkowski** distance.
2. **Select Nearest Neighbors**: Identify the $k$ closest training instances based on these distances.
3. **Make Predictions**:
   - **Classification**: Predict the class that appears most frequently among the $k$ neighbors.
   - **Regression**: Predict by averaging the target values of the $k$ neighbors.

Classification prediction example:

$$
\hat{y} = \text{majority\_vote}(y_1, y_2, \dots, y_k)
$$

---

## 📋 Key Assumptions  

- **Local Similarity**: Points close in the feature space share similar characteristics and labels.
- **Meaningful Distance Metric**: The chosen distance metric should accurately represent the true similarity or dissimilarity between data points.
- **Relevance of Features**: Assumes features are scaled and relevant, as irrelevant or noisy features can negatively impact performance.

---

## ✅ Advantages  

- **Simplicity and Interpretability**: Easy to understand, implement, and visually explain.
- **Non-linear Decision Boundaries**: Naturally adapts to complex, non-linear patterns in data.
- **No Explicit Training Phase** ("Lazy Learning"): Training is instantaneous; computational work happens at prediction time.
- **Effective with Smaller Datasets**: Performs particularly well when datasets are moderate in size and exhibit irregular decision boundaries.

---

## ❌ Disadvantages (and Mitigations)

- **Computational Cost (Prediction)**: High computational complexity for predictions, especially with large datasets.
  - *Mitigation*: Utilize optimized data structures (e.g., KD-Trees, Ball Trees, Approximate Nearest Neighbor search algorithms).

- **Sensitivity to Feature Scaling and Noise**: Distances heavily influenced by irrelevant or differently scaled features.
  - *Mitigation*: Apply feature scaling (standardization or normalization) and feature selection or dimensionality reduction (e.g., PCA).

- **Curse of Dimensionality**: Performance deteriorates as feature dimensionality increases, with distances becoming less meaningful.
  - *Mitigation*: Use dimensionality reduction methods (e.g., PCA, t-SNE, UMAP) or select relevant features to reduce complexity.

- **Selection of Hyperparameter $k$**: Poor choice of $k$ leads to either overfitting (small $k$) or underfitting (large $k$).
  - *Mitigation*: Optimize  k$ through cross-validation to ensure balanced bias-variance trade-off.

- **Memory Intensive**: Entire training dataset must be retained in memory, posing issues for large-scale data.
  - *Mitigation*: Employ data reduction strategies, instance selection methods, or approximate nearest neighbor algorithms to reduce memory usage.

---
