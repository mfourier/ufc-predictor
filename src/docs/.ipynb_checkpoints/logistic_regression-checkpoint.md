# 🔹 Model: Logistic Regression
## Assumptions, Advantages, and Disadvantages of Logistic Regression

🔹 **Overview**  
* **Logistic Regression** is a **linear model** used for **binary classification**.  
* It estimates the probability that a given input belongs to the positive class using the **logistic (sigmoid)** function.

🔹 **Mathematical Formulation**  
* Logistic regression models the **log-odds** of the probability of the positive class as a linear function of the input features:

  $$
  \log\left(\frac{P(Y=1 | X)}{1 - P(Y=1 \mid X)}\right) = w^\top X + \beta
  $$

  - $ P(Y=1 | X) $ is the predicted probability of the positive class (e.g., Fighter Blue wins).
  - $ w $ is the vector of feature weights, and $ \beta $ is the bias (intercept).
  - This implies a **linear decision boundary** in the feature space.

🔹 **Key Assumptions**  
* **Linearity in the Log-Odds**: Assumes a linear relationship between input features and the log-odds of the target.
* **Independence of Features**: Assumes features are not highly correlated (i.e., no multicollinearity).
* **No Extreme Outliers**: Outliers can heavily influence the estimation of model coefficients.
* **Sufficient Sample Size**: Requires enough samples per feature to ensure stable coefficient estimates.

🔹 **Advantages**  
* ✅ **Interpretable**: Coefficients represent the effect of a unit change in each feature on the log-odds.
* ✅ **Probabilistic Output**: Unlike models like SVM, logistic regression outputs class probabilities.
* ✅ **Efficient**: Fast to train and evaluate, even on large datasets.
* ✅ **Baseline Model**: Serves as a solid benchmark before trying more complex models.
* ✅ **Regularization Support**: Easily extended with L1 (Lasso) or L2 (Ridge) regularization to prevent overfitting.

🔹 **Disadvantages**  
* ❌ **Linear Assumption**: Struggles with non-linear relationships between features and output.
  * *Mitigation*: Use non-linear models like Decision Trees, Neural Networks, or Polynomial Feature Expansion.
* ❌ **Multicollinearity**: Highly correlated features can inflate standard errors of coefficients and destabilize the model.
  * *Mitigation*: Use **regularization** (e.g., L2), or remove/reduce correlated features using PCA or VIF.
* ❌ **Sensitive to Outliers**: Outliers can disproportionately influence the decision boundary.
  * *Mitigation*: Apply robust scaling or remove extreme outliers during preprocessing.
* ❌ **Not Suitable for Complex Decision Boundaries**: Performance degrades when the true relationship is highly non-linear or interaction-heavy.
  * *Mitigation*: Use interaction terms, polynomial features, or switch to non-linear models.
 
🔹 **Hyperparameter Tuning**  

- **Regularization Strength (C)**:  
  In scikit-learn, `C` is the **inverse** of the regularization strength.  
  - **High C** → weaker regularization (model fits data more closely),
  - **Low C** → stronger regularization (simpler model, potentially less overfitting).  
  Default is `C=1.0`.

- **Penalty Type (`penalty`)**:  
  Controls the type of regularization applied to the coefficients:
  - `'l2'`: Ridge regularization (default),
  - `'l1'`: Lasso (sparse models),
  - `'elasticnet'`: Combination of L1 and L2 (requires `solver='saga'`),
  - `'none'`: No regularization (can overfit).

- **Solver**:  
  Optimization algorithm used to fit the model. The choice of solver affects support for penalties:
  - `'liblinear'`: good for small datasets; supports L1 and L2,
  - `'saga'`: scalable to large datasets; supports all penalties including elastic net,
  - `'lbfgs'` or `'newton-cg'`: efficient for L2 regularization, not L1.

- **Class Weight (`class_weight`)**:  
  Used to handle **imbalanced datasets**. Options:
  - `'balanced'`: automatically adjusts weights inversely proportional to class frequencies,
  - `{0: w0, 1: w1}`: manually specify custom class weights.



