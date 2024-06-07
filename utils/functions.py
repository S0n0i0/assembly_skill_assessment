import json


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
