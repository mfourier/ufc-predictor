### 🔹 Model: Logistic Regression
#### Assumptions, Advantages, and Disadvantages of Logistic Regression

🔹 **Linear Relationship Between Features and Log-Odds**:
* Logistic regression assumes a linear relationship between features and the log-odds of the outcome:
  
  $$log\left(\frac{P(Y=1 | X)}{1 - P(Y=1 | X)}\right) = w^T X + \beta$$
  
* $P(Y=1 | X)$ is the probability of the positive class (e.g., fighter B wins), and $w, \beta$ are the model coefficients.
* Assumes feature independence—multicollinearity (high correlation between features) can distort the model's coefficients.

🔹 **Probability Prediction**:
* Unlike models like SVM, logistic regression predicts probabilities, offering more flexibility with decision thresholds.

🔹 **Disadvantages**:
* **Linear Assumption**: Logistic regression assumes a linear relationship between features and log-odds, which can be limiting if the true relationship is non-linear.
  * **Solution**: Use models like Decision Trees, Neural Networks, Non-linear SVM, or Polynomial Regression for complex relationships.
* **Multicollinearity**: High correlation between features can destabilize coefficients and lead to poor model performance.
  * **Solution**: Regularization or removing highly correlated features can mitigate this issue.