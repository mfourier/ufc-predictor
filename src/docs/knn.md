### 🔹 Model: K-Nearest Neighbors (KNN)
#### Assumptions, Advantages, and Disadvantages of K-Nearest Neighbors (KNN)

🔹 **Basic Concept of KNN**:
* KNN is a non-parametric, instance-based algorithm used for classification and regression.
* It classifies a test instance based on the majority class (for classification) or average (for regression) of its "k" closest training instances, using a distance metric like Euclidean distance.
* KNN doesn’t explicitly train a model but memorizes the training data, performing classification/regression when queried.

🔹 **How KNN Works**:
* KNN calculates distances between a new data point and all points in the training set.
* It selects the "k" nearest neighbors, assigning the majority class or average of the nearest values to the new point.

    $$y^​=\operatorname{majority\_vote}{({y_1​,y_2​,...,y_k​})}$$
Where $y_i$ is the class label of the i-th nearest neighbor.

🔹 **Advantages**:
* **Simple and Intuitive**: KNN is easy to implement and understand, making predictions based on similarity between data points.
* **No Training Phase**: KNN doesn’t require training; it simply stores the dataset and performs calculations when needed.
* **Flexible Decision Boundaries**: KNN handles complex patterns and works well with data having intricate decision boundaries.

🔹 **Disadvantages**:
* **Computational Complexity**: KNN is slow on large datasets due to the need to compute distances for every prediction.
  * **Solution**: Use KD-Tree or Ball-Tree data structures to speed up nearest neighbor search.
* **Sensitivity to Irrelevant Features**: Irrelevant features distort distance metrics and can degrade performance.
  * **Solution**: Apply feature selection or dimensionality reduction (e.g., PCA).
* **Choice of k**: The value of "k" influences performance—too small can lead to overfitting, too large to underfitting.
  * **Solution**: Use cross-validation to find the optimal "k".
* **Memory Intensive**: KNN requires storing the entire training dataset, which can be costly for large datasets.

