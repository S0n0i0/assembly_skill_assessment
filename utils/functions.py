import json
import numpy as np

from utils.enums import SensorMode


def load_dump(data_path):
    """Load data from JSON files.

    Args:
        data_path (str): Path to data in a JSON file.

    Returns:
        dict: data from JSON
    """

    if data_path is None:
        raise FileNotFoundError

    with open(data_path, "r") as f:
        return json.load(f)


def get_derived_SM(*modes: SensorMode) -> SensorMode:
    value_modes = [mode.value if mode is not None else None for mode in modes]

    return None if None in value_modes else max(value_modes)


# Funzione personalizzata per gestire la serializzazione di numpy float32
def np_default(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))


def compute_distance(p1: tuple[float, float], p2: tuple[float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
