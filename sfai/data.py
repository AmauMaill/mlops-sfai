from typing import List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class NaNBinarizer(BaseEstimator, TransformerMixin):
    """Make new column with True if NaN."""

    def __init__(self, columns: List = None):
        self.columns = columns

    def __str__(self):
        return f"<NaNBinarizer for {self.columns}>"

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Transform the given columns by setting True if NaN or False else.

        Args:
            X (pd.DataFrame): Features.
            y (pd.DataFrame, optional): Target. Defaults to None.

        Returns:
            pd.DataFrame: Transformed data.
        """
        cols_to_transform = X.columns

        if self.columns:
            cols_to_transform = self.columns

        if isinstance(X, pd.DataFrame):
            X[cols_to_transform] = X[cols_to_transform].isna()
        else:
            # np.isnan(X[cols_to_transform])
            raise (NotImplementedError)

        return X


def get_data_splits(X: pd.DataFrame, y: pd.DataFrame, train_size: float = 0.7) -> Tuple:
    """Generate stratified data splits by year.

    Args:
        X (pd.DataFrame): The data with the features.
        y (pd.DataFrame): The data with the target variable.
        train_size (float, optional): Size of the train set. Defaults to 0.7.

    Returns:
        Tuple: The train and test sets for the features data and the target data.
    """
    assert X.shape[0] == y.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    return X_train, X_test, y_train, y_test
