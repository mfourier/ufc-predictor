# 🔹 Model: Random Forest
## Assumptions, Advantages, Disadvantages & Hyperparameter Tuning

🔹 **Overview**  
* **Random Forest** is an **ensemble learning algorithm** used for both classification and regression tasks.  
* It operates by building multiple **decision trees**, each trained on a **random subset of the data and features**, and then aggregates their predictions:
  - **Classification** → majority vote,
  - **Regression** → average of predictions.

🔹 **How It Works**  
- **Bagging (Bootstrap Aggregating)**:  
  Each tree is trained on a different **random sample (with replacement)** of the training data. This reduces **variance** and improves generalization.
  
- **Feature Randomness**:  
  At each split in a tree, a **random subset of features** is considered rather than all. This decorrelates the trees and enhances model robustness.

- **Aggregation**:  
  The final prediction is made by **averaging (regression)** or **voting (classification)** the predictions from all trees in the ensemble.

🔹 **Key Assumptions**  
* Assumes that combining many low-bias, high-variance models (trees) will reduce variance and result in a stronger model.
* Assumes no strong linearity or feature independence; works well with **non-linear and high-dimensional** data.

---

## 🔧 Hyperparameter Tuning

| Parameter | Description | Typical Effect |
|----------|-------------|----------------|
| `n_estimators` | Number of trees in the forest | ↑ Improves performance but ↑ training time |
| `max_depth` | Maximum depth of each tree | ↓ Prevents overfitting if too deep |
| `min_samples_split` | Minimum samples required to split a node | ↑ Makes tree more conservative |
| `min_samples_leaf` | Minimum samples required at a leaf node | ↑ Smooths model; avoids tiny leaves |
| `max_features` | Number of features considered at each split | Controls diversity among trees |
| `bootstrap` | Whether to use bootstrapped datasets | `True` helps generalization |
| `class_weight` | Useful for imbalanced classification | `'balanced'` adjusts for label frequency |

📝 **Tuning Tips**:  
- Start with `n_estimators=100` or higher for stability.  
- Use `GridSearchCV` or `RandomizedSearchCV` to optimize parameters.  
- For large datasets, set `n_jobs=-1` to enable parallel training.

---

🔹 **Advantages**

* ✅ **High Accuracy**: Typically outperforms single models, especially on complex data.
* ✅ **Robust to Overfitting**: Bagging and randomness reduce variance effectively.
* ✅ **Handles Non-linearity**: Captures complex, non-linear patterns naturally.
* ✅ **Feature Importance**: Offers a ranking of feature relevance.
* ✅ **Handles Missing Data**: Can handle missing values internally.

🔹 **Disadvantages**

* ❌ **Computationally Intensive**: Training hundreds of trees takes time and memory.  
  * *Mitigation*: Use parallel processing (`n_jobs=-1`) or limit `n_estimators`, `max_depth`.

* ❌ **Reduced Interpretability**: Unlike single decision trees, the ensemble is harder to interpret.  
  * *Mitigation*: Use **feature importance** or **tree surrogates** for interpretation.

* ❌ **Memory Usage**: Each tree is stored in memory, which can be costly for large forests.  
  * *Mitigation*: Use pruning or restrict tree depth and number.

---

✅ **Use Random Forest When**:
- You want a strong baseline that performs well on most tasks.
- You don’t require interpretability as a priority.
- Your data has non-linear relationships or many features.

❌ **Avoid When**:
- You need real-time predictions with low latency.
- Your dataset is too large to fit many trees in memory.

