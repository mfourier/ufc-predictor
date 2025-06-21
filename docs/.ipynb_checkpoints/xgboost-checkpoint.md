# 🔹 XGBoost Classifier

## 📌 Overview  

**XGBoost (Extreme Gradient Boosting)** is a highly efficient and scalable boosting model based on decision trees. It is widely used in machine learning competitions due to its high accuracy, speed, and support for regularization. It is ideal for binary classification tasks such as predicting outcomes in UFC fights.

---

## 🌳 Core Concept  

XGBoost implements a **gradient boosting** approach in which sequential models are trained to correct the prediction errors of previous models. Each new tree is trained to minimize a specific loss function using gradients.

Mathematically, the model aims to minimize the following objective function:

$$
\mathcal{L}(\phi) = \sum_{i} l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)
$$

where:
- $l$ is the loss function (e.g., log loss),
- $\hat{y}_i$ is the model’s prediction,
- $\Omega(f_k)$ is a regularization term over the trees.

---

## ✅ Advantages  

- **High predictive performance** due to gradient boosting.
- **Built-in regularization** (L1 and L2) that reduces overfitting.
- **Compatible with scikit-learn**, allowing integration with pipelines, `GridSearchCV`, etc.
- **Fast and efficient training** thanks to parallelization.
- **Automatic handling of missing values and sparse data**.

---

## ❌ Disadvantages (and Mitigations)

- **Sensitive to poorly tuned hyperparameters**.
  - *Mitigation*: Perform careful tuning with cross-validation.
- **More complex to interpret** compared to linear models.
  - *Mitigation*: Use tools like SHAP for interpretability analysis.
- **Risk of overfitting** on small or noisy datasets.
  - *Mitigation*: Adjust regularization and limit tree depth.

---

## 🔧 Hyperparameters for Tuning

| Hyperparameter           | Description                                                               | Effect                                                             |
|--------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------|
| `n_estimators`           | Number of trees to train.                                                  | More trees increase capacity but also the risk of overfitting.      |
| `learning_rate`          | Step size used to update weights at each iteration.                        | Smaller rates generalize better but require more trees.             |
| `max_depth`              | Maximum depth of each tree.                                                | Increases model complexity; be cautious of overfitting.             |
| `subsample`              | Fraction of training samples used per tree.                               | Helps prevent overfitting.                                          |
| `colsample_bytree`       | Fraction of features randomly selected for each tree.                     | Similar to `subsample`, but applies to columns/features.            |
| `gamma`                  | Minimum loss reduction required to make a further partition on a leaf.    | Higher `gamma` makes the model more conservative.                   |
| `reg_alpha`, `reg_lambda`| L1 and L2 regularization parameters, respectively.                         | Control model complexity and reduce overfitting.                    |

---

## 🛠️ Example `param_grid` for GridSearchCV

```python
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
