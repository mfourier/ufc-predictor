# 🔹 Logistic Regression

## 📌 Overview  

**Logistic Regression** is a fundamental **linear classification model** widely used for binary classification problems. It calculates the probability of an input belonging to a particular class by applying the **logistic (sigmoid)** function to a linear combination of input features.

---

## 🧮 Mathematical Formulation  

Logistic regression models the **log-odds** (logit) of the probability of belonging to the positive class as follows:

$$
\log\left(\frac{P(Y=1 \mid X)}{1 - P(Y=1 \mid X)}\right) = w^\top X + \beta
$$

- $ P(Y=1 \mid X) $ is the probability that the sample belongs to the positive class (e.g., Fighter Blue winning).
- $ w $ is the weight vector associated with input features.
- $ \beta $ is the bias or intercept term.

This formulation leads to a **linear decision boundary** in the feature space.

---

## 📋 Key Assumptions

- **Linearity of Log-Odds**: Assumes a linear relationship between features and the log-odds of the target.
- **Feature Independence**: Features should ideally not exhibit high multicollinearity.
- **Minimal Outliers**: Outliers strongly affect coefficient estimates.
- **Adequate Sample Size**: Requires sufficient observations per feature for stable coefficient estimation.

---

## ✅ Advantages

- **Interpretability**: Easy-to-understand coefficients indicating the impact of each feature.
- **Probabilistic Predictions**: Directly outputs probabilities for decision-making.
- **Computational Efficiency**: Fast to train and deploy, making it suitable for large datasets.
- **Robust Baseline**: Ideal first-choice model to establish a performance baseline.
- **Regularization Capabilities**: Supports L1 (Lasso) and L2 (Ridge) regularization to mitigate overfitting.

---

## ❌ Disadvantages (and Mitigations)

- **Limited to Linear Boundaries**: Poor performance on non-linear relationships.
  - *Mitigation*: Introduce polynomial or interaction terms, or switch to non-linear models (Decision Trees, Neural Networks).
- **Multicollinearity Sensitivity**: Correlated features can destabilize coefficient estimates.
  - *Mitigation*: Apply regularization (L2), PCA, or remove correlated features using Variance Inflation Factor (VIF).
- **Outlier Sensitivity**: Decision boundaries are strongly influenced by extreme values.
  - *Mitigation*: Robust data scaling or outlier removal strategies during preprocessing.
- **Complex Decision Boundaries Limitation**: Struggles with complex, interaction-heavy data distributions.
  - *Mitigation*: Use polynomial or interaction features, or select non-linear algorithms.

---

## 🔧 Hyperparameters for Tuning

- **Regularization Strength (`C`)**: Controls the inverse of regularization intensity.
  - **High `C`** (e.g., `C=10`) → weak regularization (more complex, flexible fit).
  - **Low `C`** (e.g., `C=0.01`) → strong regularization (simpler, less prone to overfitting).

- **Penalty (`penalty`)**: Determines regularization type.
  - `'l2'` (default): Ridge regularization (shrinks coefficients towards zero without eliminating them).
  - `'l1'`: Lasso regularization (sparsity, some coefficients set exactly to zero).
  - `'elasticnet'`: Combines L1 and L2 penalties (requires solver `saga`).
  - `'none'`: No regularization, risk of overfitting.

- **Solver (`solver`)**: Optimization algorithm choice; affects penalty support.
  - `'liblinear'`: Good for smaller datasets; supports `'l1'` and `'l2'`.
  - `'lbfgs'` or `'newton-cg'`: Efficient for larger datasets with `'l2'` penalty.
  - `'saga'`: Scalable, supports `'elasticnet'`, `'l1'`, and `'l2'`; suitable for large-scale data.

- **Class Weight (`class_weight`)**: Manages imbalanced data by adjusting class importance.
  - `'balanced'`: Automatically adjusts inversely proportional to class frequencies.
  - Manual weighting: e.g., `{0: w0, 1: w1}` for explicit weight control.

---
