# 🔹 Model: K-Nearest Neighbors (KNN)
## Assumptions, Advantages, and Disadvantages of K-Nearest Neighbors

🔹 **Overview**  
* **K-Nearest Neighbors (KNN)** is a **non-parametric**, **instance-based** learning algorithm used for both **classification** and **regression** tasks.
* It makes predictions based on the similarity between the input sample and its nearest neighbors in the training data, without making strong assumptions about the underlying data distribution.

🔹 **How KNN Works**  
1. **Distance Calculation**: For a given test instance, compute the distance to all training instances using a metric like **Euclidean**, **Manhattan**, or **Minkowski** distance.
2. **Neighbor Selection**: Identify the \( k \) closest data points in the training set.
3. **Prediction**:  
   - **Classification**: Assign the majority class among the \( k \) neighbors.
   - **Regression**: Predict the average value of the \( k \) neighbors.

   $$
   \hat{y} = \operatorname{majority\_vote}(y_1, y_2, ..., y_k)
   $$

🔹 **Assumptions**  
* **Locality Matters**: Points that are close in feature space are likely to have similar target values.
* **Noisy Features Can Be Harmful**: The algorithm assumes that the distance metric meaningfully reflects similarity, which may not hold if there are irrelevant or unscaled features.

🔹 **Advantages**  
* ✅ **Simple and Intuitive**: Easy to implement, explain, and visualize.
* ✅ **No Training Required**: The model "learns" at prediction time (also called "lazy learning").
* ✅ **Flexible Decision Boundaries**: Can adapt to non-linear boundaries naturally.
* ✅ **Works Well with Small Datasets**: Especially effective when the decision boundary is irregular and the dataset is not too large.

🔹 **Disadvantages**  
* ❌ **Computationally Expensive**: High prediction cost due to distance calculation to all training samples.
  * *Mitigation*: Use efficient search structures (e.g., KD-Tree, Ball Tree, Approximate Nearest Neighbors).
* ❌ **Sensitive to Feature Scale and Irrelevant Features**: Distances can be dominated by irrelevant or unscaled features.
  * *Mitigation*: Standardize features and apply feature selection or dimensionality reduction (e.g., PCA).
* ❌ **Curse of Dimensionality**: In high dimensions, all points tend to be equidistant, reducing model performance.
  * *Mitigation*: Use dimensionality reduction techniques to reduce feature space.
* ❌ **Choice of \( k \)**: Too small → overfitting; too large → underfitting.
  * *Mitigation*: Use cross-validation to tune \( k \) based on validation performance.
* ❌ **Memory Usage**: Requires storing the full training set in memory, which is inefficient for large datasets.



