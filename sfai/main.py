import json
import tempfile
import warnings
from argparse import Namespace
from asyncio.log import logger
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from config import config
from sfai import train, utils

warnings.filterwarnings("ignore")


def etl_data():
    """Extract, load and transform data."""
    # Extract
    data = utils.load_csv_from_url(url=config.DATA_URL)

    # Transform
    data.dropna(subset=["price"], how="any", inplace=True)  # put elsewhere?

    # Load
    data_fp = Path(config.DATA_DIR, "data.json")
    utils.save_dict(d=data.to_dict(orient="records"), filepath=data_fp)

    logger.info("ETL on data is complete!")


def train_model(args_fp, experiment_name, run_name):
    """Train a model given arguments."""
    # Load data
    data_fp = Path(config.DATA_DIR, "data.json")
    data_dict = utils.load_dict(filepath=data_fp)
    data = pd.DataFrame(data_dict)

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pipeline = train.make_pipeline(args=args)
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        artifacts = train.train(args=args, data=data, pipeline=pipeline)
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))

        # Log performance and args
        mlflow.log_metrics({"mae": performance["mae"]})
        mlflow.log_metrics({"mse": performance["mse"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Export artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["pipeline"], Path(dp, "pipeline.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

        # Save to config
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))

    logger.info("Train on data is complete!")


def load_artifacts(run_id):
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    pipeline = joblib.load(Path(artifacts_dir, "pipeline.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {"args": args, "pipeline": pipeline, "performance": performance}


if __name__ == "__main__":

    args_fp = Path(config.CONFIG_DIR, "args.json")
    train_model(args_fp, experiment_name="baselines", run_name="linear_regression")
