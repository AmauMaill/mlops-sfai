import json
import random

import numpy as np
import pandas as pd


def load_csv_from_url(url):
    """Load csv data from a URL."""
    data = pd.read_csv(url)
    return data


def load_dict(filepath):
    """Load a dictionary from a JSON's filepath."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d, filepath, cls=None, sortkeys=False):
    """Save a dictionary to a specific location."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seeds(seed=42):
    """Set seeds for repoducibility."""
    np.random.seed(seed)
    random.seed(seed)
