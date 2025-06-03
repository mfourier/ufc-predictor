import pandas as pd
from typing import Tuple, Optional, List, Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class UFCData:
    """
    Manages and preprocesses UFC fight datasets for the UFC Fight Predictor machine learning pipeline.

    Features:
        - Validates data integrity and checks for missing values.
        - Splits the dataset into stratified training and test sets.
        - Automatically detects categorical, binary, multiclass, and numerical columns.
        - Standardizes numerical features using training set statistics.
        - Encodes categorical variables for modeling.
        - Provides access to both raw and processed train/test splits.
        - Summarizes dataset composition.

    Attributes:
        raw_df (pd.DataFrame): The original input DataFrame, including features and 'label' column.
        test_size (float): Proportion of the dataset used as test data.
        categorical_columns (List[str]): Columns automatically detected as categorical (dtype 'object', 'category', or 'bool').
        numerical_columns (List[str]): Columns detected as numerical (all columns not categorical).
        binary_columns (List[str]): Categorical columns with exactly two unique non-null values.
        multiclass_columns (List[str]): Categorical columns with more than two unique values.
        _X_train, _X_test (pd.DataFrame): Feature sets for training and testing.
        _y_train, _y_test (pd.Series): Target labels for training and testing.
        _X_train_processed, _X_test_processed (Optional[pd.DataFrame]): Processed feature matrices (standardized and encoded).
        _scaler (StandardScaler | MinMaxScaler | RobustScaler | None): Fitted scaler instance used for standardization.
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

        X = self.raw_df.drop(columns='label')
        y = self.raw_df['label']

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.numerical_columns = [col for col in X.columns if col not in self.categorical_columns]
        
        self.binary_columns = [col for col in self.categorical_columns if X[col].dropna().nunique() == 2]
        self.multiclass_columns = [col for col in self.categorical_columns if col not in self.binary_columns]

        self._X_train_processed: Optional[pd.DataFrame] = None
        self._X_test_processed: Optional[pd.DataFrame] = None
        self._scaler = None

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

    def summary(self, show_columns: int = 5) -> None:
        print(f"\n\U0001F4CA UFCData Summary:")
        print(f"   - Total samples       : {self.raw_df.shape[0]}")
        print(f"   - Train samples       : {self._X_train.shape[0]}")
        print(f"   - Test samples        : {self._X_test.shape[0]}")
        print(f"   - Feature columns     : {self._X_train.shape[1]}")
        print(f"   - Columns preview     : {', '.join(self._X_train.columns[:show_columns])}...\n")
