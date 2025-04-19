# 🔹 Model: Support Vector Machine (SVM)
## Assumptions, Advantages, and Disadvantages of Support Vector Machines (SVM)

🔹 **Core Idea**  
Support Vector Machine (SVM) is a **supervised learning algorithm** used for **classification** and **regression** tasks (SVR). Its main goal is to find the **optimal separating hyperplane** that **maximizes the margin** between classes.

* The **margin** is the distance between the decision boundary and the nearest data points from each class.
* These closest data points are called **support vectors**, and they are the only points that influence the final model.

🔹 **Linear SVM**  
For linearly separable data, SVM finds the hyperplane $ y = w^T x + b $ that **maximizes the margin** between classes:

- $ w $: weight vector (perpendicular to the hyperplane),
- $ b $: bias (offset from origin),
- Margin is $ \frac{2}{\|w\|} $, and the goal is to minimize $ \|w\|^2 $ under correct classification constraints.

🔹 **Nonlinear SVM & the Kernel Trick**  
When the data is not linearly separable, SVM uses the **kernel trick** to implicitly map data to a **higher-dimensional space**, where a linear separator **can** exist.

* Common kernels:
  - **Linear Kernel**: $ K(x, x') = x^T x' $
  - **Polynomial Kernel**: $ K(x, x') = (\gamma x^T x' + r)^d $
  - **RBF (Radial Basis Function)**: $ K(x, x') = \exp(-\gamma \|x - x'\|^2) $

* The kernel trick allows SVM to operate in high-dimensional spaces without the computational cost of explicitly computing the transformation.

🔹 **Advantages**

✅ **Effective in High-Dimensional Spaces**  
Handles large feature sets well, making it suitable for text classification and genomic data.

✅ **Robust Generalization**  
Maximizing the margin encourages better generalization and reduces the risk of overfitting (especially with proper C).

✅ **Versatile**  
With kernel functions, SVM can model both linear and complex nonlinear relationships.

🔹 **Disadvantages**

❌ **Computationally Intensive**  
Training time grows significantly with the size of the dataset (especially for non-linear kernels).  
→ *Solution*: Use **LinearSVC** or **SGDClassifier** for large-scale problems.

❌ **Sensitive to Hyperparameters**  
The choice of kernel, C, and γ can greatly affect performance.  
→ *Best Practice*: Use **grid search + cross-validation** to tune.

❌ **Memory Usage**  
Support vectors must be stored for prediction, which can become memory-intensive for large datasets.

🔹 **Assumptions**

* No strong statistical assumptions (e.g., feature independence).
* Assumes that data is **somewhat separable** (linearly or nonlinearly).
* Best performance is achieved when classes are well-separated with few overlapping points.

🔹 **Use Cases**

* ✅ **Text Classification (e.g., spam detection, sentiment analysis)**
* ✅ **Image Classification**
* ✅ **Bioinformatics and Genomics**
* ✅ **Any high-dimensional, low-sample-size problems**

🔹 **Hyperparameter Tuning**  

- **Regularization Parameter (C)**:  
  Balances the trade-off between a wide margin and training accuracy.
  - **High C** → fewer misclassifications, smaller margin (hard margin).
  - **Low C** → allows more misclassifications, larger margin (soft margin).

- **Kernel Type**:  
  Determines the transformation of data. Choosing the right kernel is crucial for performance.

- **Gamma (γ)** *(only for RBF or polynomial kernels)*:  
  Controls the influence of a single training point.  
  - **High γ** → close influence (may overfit),
  - **Low γ** → broader influence (may underfit).
