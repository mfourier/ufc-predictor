# 🔹 Quadratic Discriminant Analysis (QDA)

## 📌 Overview  

**Quadratic Discriminant Analysis (QDA)** is a generative classification method that models each class separately by assuming features follow class-specific multivariate Gaussian distributions. Unlike Linear Discriminant Analysis (LDA), QDA allows each class to have its own covariance matrix, resulting in **quadratic decision boundaries**, making it more flexible for certain types of data.

---

## 🧮 How QDA Works  

QDA estimates the probability of a given input vector $x$ belonging to class $k$ by assuming a multivariate Gaussian distribution for each class:

$$
P(X = x \mid Y = k) = \frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k)\right)
$$

- $\mu_k$ is the mean vector of class $k$.
- $\Sigma_k$ is the covariance matrix specific to class $k$.
- $d$ is the dimensionality of the feature space.

The predicted class is the one maximizing the posterior probability:

$$
\hat{y} = \arg\max_k P(Y = k \mid X = x)
$$

---

## 📋 Key Assumptions

- **Gaussian Distributed Features**: Assumes that features within each class follow a Gaussian (normal) distribution.
- **Class-specific Covariance Matrices**: Each class can have its own covariance matrix, enabling non-linear (quadratic) decision boundaries.
- **Independent and Identically Distributed (IID)** samples: Assumes data points are independently sampled from class-specific distributions.

---

## ✅ Advantages

- **Flexible Decision Boundaries**: Can capture quadratic (curved) decision boundaries, accommodating complex class separations better than linear methods.
- **Class-specific Covariance**: More robust in scenarios where class distributions vary significantly.
- **Probabilistic Predictions**: Provides direct probability estimates of class membership, facilitating uncertainty quantification.
- **Computational Efficiency**: Faster to train and predict compared to non-parametric methods (e.g., KNN), especially on moderate-sized datasets.

---

## ❌ Disadvantages (and Mitigations)

- **Sensitivity to Gaussian Assumption**: Poor performance if data significantly deviates from Gaussian distributions.
  - *Mitigation*: Apply feature transformations, verify distribution assumptions through exploratory analysis, or consider non-parametric alternatives.

- **High-dimensional Challenges**: Covariance estimation becomes difficult as dimensionality increases (risk of singular covariance matrices).
  - *Mitigation*: Use dimensionality reduction (e.g., PCA) or regularization approaches to stabilize covariance estimates.

- **Overfitting Risk**: Greater flexibility can lead to overfitting when sample sizes are small relative to the feature dimension.
  - *Mitigation*: Regularize covariance matrices, reduce feature dimensionality, or ensure sufficient sample sizes per class.

- **Sensitivity to Outliers**: Outliers disproportionately influence the estimation of class covariances and means.
  - *Mitigation*: Employ robust covariance estimators, remove or correct extreme outliers during preprocessing.

---

## 🔧 Hyperparameters and Considerations  

QDA generally has fewer hyperparameters, but key considerations include:

| Parameter                  | Description                                         | Effect on Model Performance              |
|----------------------------|-----------------------------------------------------|-------------------------------------------|
| `priors`                   | Class prior probabilities (can be user-defined or inferred from data). | Adjusting priors affects class bias and can mitigate imbalance. |
| `regularization`           | Techniques to stabilize covariance estimation (e.g., covariance shrinkage). | Reduces overfitting and stabilizes performance in high-dimensional settings. |

---

## 📝 Best Practices for QDA  

- **Validate Gaussian Assumption**: Always inspect your data distribution before applying QDA.
- **Regularization in High Dimensions**: Employ covariance regularization when the number of features is large relative to available samples.
- **Data Scaling**: Standardize or normalize data to reduce sensitivity to feature scales and improve stability in covariance estimation.
- **Cross-Validation**: Use cross-validation to reliably evaluate model performance and to tune regularization strategies.

---
