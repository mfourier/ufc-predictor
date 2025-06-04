import os
import re
import math
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

save_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../img/")
)

class UFCData:
    """
    Manages and preprocesses UFC fight datasets for ML pipelines.

    Features:
        - Validates data integrity (e.g., missing values, required columns).
        - Splits data into stratified training/testing sets.
        - Automatically categorizes columns (categorical, numerical, binary, multiclass).
        - Applies feature scaling and categorical encoding.
        - Computes and visualizes correlation matrices.

    Attributes:
        raw_df (pd.DataFrame): Original input data.
        test_size (float): Proportion of dataset for testing.
        categorical_columns (List[str]): Detected categorical features.
        numerical_columns (List[str]): Detected numerical features.
        binary_columns (List[str]): Detected binary categorical features.
        multiclass_columns (List[str]): Detected multiclass categorical features.
        _scaler (Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]): Scaler instance used for feature scaling.
        _X_train_processed (Optional[pd.DataFrame]): Preprocessed training features.
        _X_test_processed (Optional[pd.DataFrame]): Preprocessed testing features.
        _corr (Optional[pd.DataFrame]): Cached correlation matrix.

    Initialization:
        Args:
            df (pd.DataFrame): DataFrame containing features and a 'label' column.
            test_size (float, optional): Fraction of data for testing. Default is 0.2.
            random_state (int, optional): Random seed for reproducibility. Default is 42.
            validate_nan (bool, optional): If True, raises error if missing values are present. Default is True.

        Raises:
            ValueError: If 'label' column is missing or data contains NaN values when `validate_nan` is True.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        validate_nan: bool = True,
        scaler_type: Literal['standard', 'minmax', 'robust'] = 'standard',
    ):
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain a 'label' column.")

        if validate_nan and df.isnull().any().any():
            raise ValueError("DataFrame contains missing values. Please clean the data.")

        self.raw_df: pd.DataFrame = df.reset_index(drop=True)
        self.test_size = test_size

        self._X = self.raw_df.drop(columns='label')
        self._y = self.raw_df['label']

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, test_size=test_size, random_state=random_state, stratify=self._y
        )

        self.categorical_columns = self._X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.numerical_columns = [col for col in self._X.columns if col not in self.categorical_columns]
        self.binary_columns = [col for col in self.categorical_columns if self._X[col].dropna().nunique() == 2]
        self.multiclass_columns = [col for col in self.categorical_columns if col not in self.binary_columns]

        self._X_train_processed: Optional[pd.DataFrame] = None
        self._X_test_processed: Optional[pd.DataFrame] = None
        self._scaler = None
        self._corr = None
        
    # ---------- Data Engineering ----------

    def standardize(self, scaler_type: Literal['standard', 'minmax', 'robust'] = 'standard') -> None:
        """
        Standardize numerical columns using training data statistics.

        Args:
            scaler_type (str): Type of scaler to use: 'standard', 'minmax', or 'robust'.
        """
        if scaler_type == 'minmax':
            self._scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self._scaler = RobustScaler()
        else:
            self._scaler = StandardScaler()

        self._X_train_processed = self._X_train.copy()
        self._X_test_processed = self._X_test.copy()

        self._X_train_processed[self.numerical_columns] = self._scaler.fit_transform(
            self._X_train[self.numerical_columns]
        )
        self._X_test_processed[self.numerical_columns] = self._scaler.transform(
            self._X_test[self.numerical_columns]
        )

    def encode(self) -> None:
        """
        Encode categorical features using pandas' get_dummies.
    
        - Binary categorical columns are encoded with `drop_first=True` to avoid multicollinearity.
        - Multiclass categorical columns are encoded with `drop_first=False` to retain all information.
        - All encoded features are aligned between train and test to ensure consistent columns.
        - Numerical features are preserved and concatenated with the encoded categorical features.
    
        After calling this method:
            - `self._X_train_processed` and `self._X_test_processed` will contain the fully encoded feature matrices,
              ready for model training and evaluation.
    
        Returns:
            None
        """
        
        X_train_base = self._X_train_processed if self._X_train_processed is not None else self._X_train.copy()
        X_test_base = self._X_test_processed if self._X_test_processed is not None else self._X_test.copy()
    
        # Encode binary categorical columns with drop_first=True
        X_train_bin = pd.get_dummies(
            X_train_base[self.binary_columns], drop_first=True
        ).astype(int)
        X_test_bin = pd.get_dummies(
            X_test_base[self.binary_columns], drop_first=True
        ).astype(int)
    
        # Encode multiclass categorical columns with drop_first=False
        X_train_multi = pd.get_dummies(
            X_train_base[self.multiclass_columns], drop_first=False
        ).astype(int)
        X_test_multi = pd.get_dummies(
            X_test_base[self.multiclass_columns], drop_first=False
        ).astype(int)
    
        X_test_bin = X_test_bin.reindex(columns=X_train_bin.columns, fill_value=0)
        X_test_multi = X_test_multi.reindex(columns=X_train_multi.columns, fill_value=0)
    
        self._X_train_processed = pd.concat([X_train_bin, X_train_multi, X_train_base[self.numerical_columns]], axis=1)
        self._X_test_processed = pd.concat([X_test_bin, X_test_multi, X_test_base[self.numerical_columns]], axis=1)

    def compute_corr(self, method: str = 'pearson', recalculate: bool = True) -> pd.DataFrame:
        """
        Compute (and optionally cache) the correlation matrix for the processed training features.

        Args:
            method (str): Correlation method to use: 'pearson', 'spearman', or 'kendall'.
            recalculate (bool): If True (default), recalculates and updates the cached correlation matrix.
                                If False, returns the cached matrix if available.

        Returns:
            pd.DataFrame: Correlation matrix (features x features) for the processed training set.

        Raises:
            ValueError: If processed training features are not available.

        Notes:
            - The matrix is cached as `self._corr`.
            - For feature analysis and leakage-free pipelines, use only training data.
        """
        if self._X_train_processed is None:
            raise ValueError(
                "Processed training features not found. Please run standardize() and encode() before calling this method."
            )
        # Only recalculate if needed or not already cached
        if recalculate or not hasattr(self, "_corr") or self._corr is None:
            self._corr = self._X_train_processed.corr(method=method)

    # ---------- Data Visualization ----------

    def plot_corr(
    self,
    threshold: float = None,
    figsize: tuple = (12, 10),
    annot: bool = False,
    cmap: str = "coolwarm",
    title: str = "Correlation Matrix (Processed Train Set)",
    fmt: str = ".2f",
    vmin: float = -1,
    vmax: float = 1,
    cbar: bool = True,
    save_file: bool = False
    ) -> None:
        """
        Plots the correlation matrix stored in self._corr.
        If threshold is set, only features with at least one correlation
        (absolute value) above the threshold (excluding the diagonal) are plotted.
    
        Args:
            threshold (float, optional): Minimum absolute correlation to include. If None, plots all.
            figsize (tuple): Figure size (width, height).
            annot (bool): Annotate coefficients on the heatmap.
            cmap (str): Matplotlib colormap for visualization.
            title (str): Plot title.
            fmt (str): Format string for annotations (e.g., '.2f').
            vmin, vmax (float): Value limits for colormap.
            cbar (bool): Whether to display colorbar.
            save_file (bool): If True, saves the figure to /ufc-predictor/img/ with an autogenerated name.
    
        Raises:
            ValueError: If correlation matrix is not computed.
        """
        if self._corr is None or not isinstance(self._corr, pd.DataFrame):
            raise ValueError(
                "Correlation matrix not found. Please run correlation_matrix() first."
            )
        corr = self._corr
    
        # Determine which columns to plot based on threshold
        if threshold is not None:
            # Mask of |corr| >= threshold (excluding diagonal)
            mask = (corr.abs() >= threshold) & (corr.abs() < 1)
            cols_to_plot = mask.any(axis=0)
            selected_cols = corr.columns[cols_to_plot].tolist()
            if not selected_cols:
                print(f"No correlations found above |{threshold}|.")
                return
            corr = corr.loc[selected_cols, selected_cols]
            title = f"Correlation Matrix (|r| ≥ {threshold})"
    
        plt.figure(figsize=figsize)
        sns.set(style="whitegrid", font_scale=1.1)
        ax = sns.heatmap(
            corr,
            annot=annot,
            cmap=cmap,
            fmt=fmt,
            linewidths=0.5,
            linecolor='gray',
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=cbar,
            annot_kws={"size": 9} if annot else None
        )
        plt.title(title, fontsize=18, weight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
    
        if save_file:
            # Construye la ruta a /img/ relativa al archivo actual
            img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../img/"))
            os.makedirs(img_dir, exist_ok=True)
            # Construye un nombre de archivo seguro y único
            # Quita caracteres no válidos del título y añade threshold
            safe_title = re.sub(r'[^\w\-_\. ]', '_', title)
            fname = f"{safe_title}"
            if threshold is not None:
                fname += f"_thr{str(threshold).replace('.','p')}"
            fname += ".png"
            save_path = os.path.join(img_dir, fname)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
    
        plt.show()
    
    def top_corr(
        self,
        threshold: float = 0.5,
        absval: bool = True,
        n: int = 50
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with the top n pairs with correlation above the threshold.
    
        Args:
            threshold (float): Minimum (absolute) correlation to include.
            absval (bool): Use absolute value of correlation.
            n (int): Maximum number of pairs to show.
    
        Returns:
            pd.DataFrame: DataFrame with columns ['Feature 1', 'Feature 2', 'Correlation']
    
        Raises:
            ValueError: If correlation matrix is not computed.
        """
        if self._corr is None:
            raise ValueError("Compute the correlation matrix first with correlation_matrix().")
        corr = self._corr.copy()
        if absval:
            corr = corr.abs()
        # Remove self-correlation (keep only upper triangle)
        mask = np.triu(np.ones(corr.shape), 1).astype(bool)
        corr_pairs = corr.where(mask).stack().reset_index()
        corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
        corr_pairs = corr_pairs.loc[corr_pairs['Correlation'] >= threshold]
        return corr_pairs.sort_values('Correlation', ascending=False).head(n)

    def plot_feature_distributions(
    self, 
    processed: bool = False, 
    max_unique: int = 12, 
    bins: int = 20, 
    features_per_fig: int = 12,   # NUEVO parámetro
    figsize: tuple = (20, 10), 
    save_file: bool = False,
    suptitle: str = None
) -> None:
        """
        Plots feature distributions as histograms (numerical) or countplots (categorical/codified)
        for either raw or processed training data. Splits features into multiple figures if needed.
    
        Args:
            features_per_fig (int): Number of features per figure page.
            ... (otros args)
        """
        import math
    
        if processed:
            if self._X_train_processed is None:
                raise ValueError("Processed training data not found. Run standardize() and encode() first.")
            df = self._X_train_processed.copy()
            title_prefix = "Processed"
        else:
            df = self._X_train.copy()
            title_prefix = "Raw"
    
        feature_list = df.columns.tolist()
        n_features = len(feature_list)
        n_pages = math.ceil(n_features / features_per_fig)
        for page in range(n_pages):
            plt.figure(figsize=figsize)
            sns.set(style="whitegrid")
            start = page * features_per_fig
            end = min((page + 1) * features_per_fig, n_features)
            cols_this_page = feature_list[start:end]
            ncols = min(4, len(cols_this_page))
            nrows = math.ceil(len(cols_this_page) / ncols)
            for idx, col in enumerate(cols_this_page, 1):
                plt.subplot(nrows, ncols, idx)
                col_data = df[col]
                unique_vals = col_data.nunique(dropna=True)
                if pd.api.types.is_numeric_dtype(col_data):
                    if unique_vals <= max_unique:
                        sns.countplot(x=col_data, hue=col_data, palette="crest", legend=False)
                    else:
                        sns.histplot(col_data, kde=True, bins=bins, color='dodgerblue')
                else:
                    sns.countplot(x=col_data, hue=col_data, palette="crest", legend=False)
                plt.title(col, fontsize=11)
                plt.xlabel("")
                plt.ylabel("Count")
                plt.xticks(rotation=35, ha='right', fontsize=9)
            plt.tight_layout()
            page_title = suptitle or f"{title_prefix} Feature Distributions (Train Set)"
            if n_pages > 1:
                page_title += f" (Page {page+1}/{n_pages})"
            plt.suptitle(page_title, fontsize=18, y=1.04, weight='bold')
            plt.subplots_adjust(top=0.9, hspace=0.35)
            plt.tight_layout()
            if save_file:
                img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../img/"))
                os.makedirs(img_dir, exist_ok=True)
                safe_title = re.sub(r'[^\w\-_\. ]', '_', page_title)
                fname = f"{safe_title}_features.png"
                save_path = os.path.join(img_dir, fname)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            plt.show()
    
    # ---------- Access to raw split ----------

    def get_X_train(self) -> pd.DataFrame:
        """Returns the raw (non-standardized) training features."""
        return self._X_train

    def get_X_test(self) -> pd.DataFrame:
        """Returns the raw (non-standardized) test features."""
        return self._X_test

    def get_y_train(self) -> pd.Series:
        """Returns the raw training labels."""
        return self._y_train

    def get_y_test(self) -> pd.Series:
        """Returns the raw test labels."""
        return self._y_test

    def get_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the raw (non-standardized) training set."""
        return self._X_train, self._y_train

    def get_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the raw (non-standardized) testing set."""
        return self._X_test, self._y_test

    def get_df_train(self) -> [pd.DataFrame, pd.Series]:
        """Returns the raw (non-standardized) concatenated data testing set."""
        X,y = self.get_train()
        return pd.concat([X, y.rename("label")], axis=1)

    def get_df_test(self) -> [pd.DataFrame, pd.Series]:
        """Returns the raw (non-standardized) concatenated data training set."""
        X,y = self.get_test()
        return pd.concat([X, y.rename("label")], axis=1)
        
    # ---------- Access to processed data split ----------
    
    def get_processed_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the standardized/encoded training set."""
        if self._X_train_processed is None:
            raise ValueError("Training data has not been standardized.")
        return self._X_train_processed, self._y_train

    def get_processed_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the standardized/encoded testing set."""
        if self._X_test_processed is None:
            raise ValueError("Test data has not been standardized.")
        return self._X_test_processed, self._y_test

    def get_df_processed_train(self) -> [pd.DataFrame, pd.Series]:
        """Returns standardized/encoded concatenated df training set."""
        X,y = self.get_processed_train()
        return pd.concat([X, y.rename("label")], axis=1)

    def get_df_processed_test(self) -> [pd.DataFrame, pd.Series]:
        """Returns standardized/encoded concatenated df testing set."""
        X,y = self.get_processed_test()
        return pd.concat([X, y.rename("label")], axis=1)
        
    def get_scaler(self):
        """Returns the fitted scaler instance used for standardization."""
        return self._scaler

    def to_dict(self) -> dict:
        """Exports all raw splits as a dictionary."""
        return {
            "X_train": self._X_train,
            "y_train": self._y_train,
            "X_test": self._X_test,
            "y_test": self._y_test
        }

    # ---------- Summary ----------

    def summary(self) -> None:
        """Print dataset summary."""
        print(f"🧪Samples: {self.raw_df.shape[0]}")
        print(f"🧪Train/Test split: {self._X_train.shape[0]}/{self._X_test.shape[0]}")
        print(f"🧪Features: {self._X_train.shape[1]}")
        print(f"🧪Categorical: {len(self.categorical_columns)}, Numerical: {len(self.numerical_columns)}")
