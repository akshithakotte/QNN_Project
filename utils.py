import os
import pickle
from typing import Any

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def save_pickle(obj: Any, name: str):
    path = os.path.join(MODELS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
