from typing import Dict

import pandas as pd


def predict(data: pd.DataFrame, artifacts: Dict) -> Dict:
    """Predict prices for given house(s).

    Args:
        data (pd.DataFrame): The input data.
        artifacts (Dict): A dictionnary where the trained pipeline is.

    Returns:
        Dict: The input data processed and the prediction(s).
    """
    transformed_data = artifacts["pipeline"].transform(data)
    y_pred = artifacts["pipeline"].predict(data)

    predictions = [
        {"input_data": transformed_data[:, i], "predicted_price": y_pred[i]}
        for i in range(len(y_pred))
    ]
    return predictions
