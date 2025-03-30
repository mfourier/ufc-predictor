### 🔹 Model: Support Vector Machine (SVM)
#### Assumptions, Advantages, and Disadvantages of Support Vector Machines (SVM)

🔹 **Basic Concept of SVM**:

Support Vector Machine (SVM) is a supervised learning algorithm primarily used for classification tasks, although it can also be used for regression (SVR). It works by finding the hyperplane that best separates data points of different classes.  
The decision boundary is defined by **support vectors**, which are the closest data points from each class. These points are critical in determining the position of the hyperplane.  
SVM uses **kernel functions** to transform data into higher dimensions, enabling the separation of non-linear data by finding linear decision boundaries in the transformed feature space.

🔹 **How SVM Works**:

- **Linear SVM**: In cases where the data is linearly separable, SVM attempts to find the hyperplane that maximizes the margin (distance) between data points of different classes.

  $$y = w^T x + b$$  
  Where $w$ represents the weight vector and $b$ is the bias term.

- **Non-Linear SVM**: When the data is not linearly separable, SVM applies a **kernel trick** to transform the data into a higher-dimensional space where a hyperplane can separate the classes. Common kernels include:
    - **Linear kernel**: For linearly separable data.
    - **Radial Basis Function (RBF)**: For non-linearly separable data.
    - **Polynomial kernel**: Used for non-linear data with a polynomial decision boundary.

  The kernel trick allows SVM to work efficiently even in high-dimensional spaces without explicitly transforming the data.

🔹 **Hyperparameter Tuning**:

- **Regularization Parameter (C)**: Controls the trade-off between maximizing the margin and minimizing classification error. A higher value of C leads to a narrower margin and fewer misclassifications, whereas a lower value encourages a wider margin with more misclassifications.
- **Kernel Type**: Defines the function used to transform the data into a higher-dimensional space. Common choices include linear, RBF, and polynomial kernels.
- **Gamma**: Determines the influence of a single training point in the RBF kernel. A small gamma results in a model that is more influenced by distant points, while a large gamma focuses more on nearby points.

🔹 **Advantages**:

- **Effective in High-Dimensional Spaces**: SVM performs well in high-dimensional spaces and is effective in cases where the number of dimensions exceeds the number of samples.
- **Robust to Overfitting**: By maximizing the margin between classes, SVM is less likely to overfit, especially when using a good regularization parameter (C).
- **Versatile with Kernels**: The use of kernel functions allows SVM to handle both linear and non-linear data effectively.

🔹 **Disadvantages**:

- **Computationally Intensive**: SVM can be computationally expensive, especially with large datasets, as it involves solving a quadratic optimization problem.
    - **Solution**: Use approximate methods like Stochastic Gradient Descent (SGD) or LinearSVC for large datasets.
- **Choice of Kernel and Hyperparameters**: The performance of SVM is sensitive to the choice of kernel and hyperparameters. Grid search and cross-validation are often required to tune these hyperparameters effectively.
- **Memory Usage**: Storing support vectors for large datasets can be memory-intensive, especially in non-linear SVM.
