#### 🔹 Performing Hyperparameter Tuning with GridSearchCV to Optimize the Models

1. **Hyperparameter Tuning with GridSearchCV:**
   * The `param_grid` dictionary defines a range of hyperparameters to tune. For example, in **K-Nearest Neighbors (KNN)**, this could include `n_neighbors`, `weights`, and `metric`.
   * **GridSearchCV** exhaustively searches over all specified hyperparameter combinations using **5-fold cross-validation** (`cv=5`).
   * After training, the best set of parameters can be accessed using `grid_search.best_params_`.

2. **Best Model Selection:**
   * After grid search completion, the best model is obtained via `grid_search.best_estimator_`, which provides the model trained with the optimal hyperparameters.

3. **Prediction and Evaluation:**
   * Predictions are made using `model.predict(X_test)` on the test set.
   * Model performance is evaluated with **accuracy** and the **classification report** (`classification_report`), which includes metrics such as **precision**, **recall**, and **F1-score**.
