import pandas as pd
from typing import Tuple, Optional, List, Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class UFCData:
    """
    UFCData encapsulates UFC fight dataset management, including:

    - Validation of data integrity
    - Train/test splitting with stratification
    - Automatic detection of categorical and numerical features
    - Standardization of numerical features based on training data
    - Storage of standardized versions of train/test splits
    - Retention of fitted scaler for future inference

    Attributes:
        raw_df (pd.DataFrame): The original full dataset including features and label.
        test_size (float): Proportion of the dataset to use as test data.
        categorical_columns (list): Automatically detected categorical columns.
        numerical_columns (list): Remaining columns considered numerical.
        _X_train, _X_test (pd.DataFrame): Split feature sets.
        _y_train, _y_test (pd.Series): Corresponding target labels.
        _X_train_processed, _X_test_processed (pd.DataFrame): Standardized feature sets.
        _scaler: Fitted scaler instance (StandardScaler, MinMaxScaler, etc.).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        validate_nan: bool = True,
        auto_standardize: bool = False,
        scaler_type: Literal['standard', 'minmax', 'robust'] = 'standard',
        columns_to_scale: Optional[List[str]] = None
    ):
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain a 'label' column.")

        if validate_nan and df.isnull().any().any():
            raise ValueError("DataFrame contains missing values. Please clean the data.")

        self.raw_df: pd.DataFrame = df.reset_index(drop=True)
        self.test_size = test_size

        X = self.raw_df.drop(columns='label')
        y = self.raw_df['label']

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numerical_columns = [col for col in X.columns if col not in self.categorical_columns]

        self._X_train_processed: Optional[pd.DataFrame] = None
        self._X_test_processed: Optional[pd.DataFrame] = None
        self._scaler = None

        if auto_standardize:
            cols = columns_to_scale if columns_to_scale else self.numerical_columns
            self.standardize(cols, scaler_type=scaler_type)

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
        """Returns the raw (non-standardized) test set."""
        return self._X_test, self._y_test

    # ---------- Processed (standardized) data ----------

    def standardize(self, columns_to_scale: List[str], scaler_type: Literal['standard', 'minmax', 'robust'] = 'standard') -> None:
        """
        Standardize numerical columns using training data statistics.

        Args:
            columns_to_scale (list[str]): Columns to apply scaling to.
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

        self._X_train_processed[columns_to_scale] = self._scaler.fit_transform(
            self._X_train[columns_to_scale]
        )
        self._X_test_processed[columns_to_scale] = self._scaler.transform(
            self._X_test[columns_to_scale]
        )

    def get_processed_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the standardized training set."""
        if self._X_train_processed is None:
            raise ValueError("Training data has not been standardized.")
        return self._X_train_processed, self._y_train

    def get_processed_test(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the standardized test set."""
        if self._X_test_processed is None:
            raise ValueError("Test data has not been standardized.")
        return self._X_test_processed, self._y_test

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

    def summary(self, show_columns: int = 5) -> None:
        print(f"\n\U0001F4CA UFCData Summary:")
        print(f"   - Total samples       : {self.raw_df.shape[0]}")
        print(f"   - Train samples       : {self._X_train.shape[0]}")
        print(f"   - Test samples        : {self._X_test.shape[0]}")
        print(f"   - Feature columns     : {self._X_train.shape[1]}")
        print(f"   - Columns preview     : {', '.join(self._X_train.columns[:show_columns])}...\n")
