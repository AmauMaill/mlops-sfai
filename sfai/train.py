from typing import Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from sfai.data import NaNBinarizer, get_data_splits


def make_pipeline(args: Dict) -> Pipeline:
    """Create the scikit learn pipeline.

    Args:
        args (Dict): A dictionnary with information to be passed in the pipeline.

    Returns:
        Pipeline: A scikit learn pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("ohe_nan", NaNBinarizer(columns=args.transform)),
            ("median_imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("linear_reg", LinearRegression()),
        ]
    )
    return pipeline


def train(args: Dict, data: pd.DataFrame, pipeline: Pipeline, trial: bool = None) -> Dict:
    """Train model on data

    Args:
        args (Dict): A dictionnary with columns for features and target.
        data (pd.DataFrame): The data passed to the pipeline.
        pipeline (Pipeline): A pipeline to train.
        trial (bool, optional): Not used here. Defaults to None.

    Returns:
        Dict: The args, pipeline and performance.
    """

    # Prepare data
    features = args.features
    target = args.target

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = get_data_splits(X, y)

    # Run pipeline
    pipeline.fit(X=X_train, y=y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)

    performance = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
    }

    return {"args": args, "pipeline": pipeline, "performance": performance}
