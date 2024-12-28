import json
import numpy as np
from itertools import chain, combinations
import os
import pickle
from functools import reduce
import operator

from utils.enums import SensorMode
from utils.classes import PathSource


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


def all_subsets(people_split, start_from=2, all_elements=True):
    return chain(
        *map(
            lambda x: combinations(people_split, x),
            range(start_from, len(people_split) + (1 if all_elements else 0)),
        )
    )


def extract_by_key(env, key, verbose=False):
    """
    input:
        env: lmdb environment initialized (see main function)
        key: the frame number in lmdb key format for which the feature is to be extracted
             the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
             e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84346135_mono10bit/HMC_84346135_mono10bit_0000000001.jpg
    output: a 2048-D np-array (TSM feature corresponding to the key)
    """
    data = env.get(key.encode("utf-8"))
    if verbose and data is None:
        print(f"[ERROR] Key {key} does not exist !!!")
    if data is None:
        return None
    else:
        return np.frombuffer(data, "float32")  # convert to numpy array


def get_offsets(offsets_path):
    offsets = {}
    with open(offsets_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        _, file, start_frame, new_end_frame = line.strip().split(",")
        sequence, view = file.split("/")
        if sequence not in offsets:
            offsets[sequence] = {}
        offsets[sequence][view[:-4]] = {
            "start_frame": int(start_frame),
            "new_end_frame": int(new_end_frame) if new_end_frame != "-" else -1,
        }
    return offsets


def get_view_frames_data(path):
    frames_data = {}
    with open(path) as offsets:
        lines = offsets.readlines()
    for line in lines[1:]:
        _, video, start_frame, new_end_frame = line.split(",")
        name, general_view = video.split("/")
        if name not in frames_data["ego"]:
            frames_data["ego"][name] = {}
        frames_data["ego"][name][general_view] = {
            "first_frame": int(start_frame),
            "new_end_frame": -1 if new_end_frame == "-" else int(new_end_frame),
        }


def get_dict_element(data: dict, keys: list):
    try:
        return reduce(operator.getitem, keys, data)
    except:
        return None


def set_dict_element(data: dict, keys: list, value):
    previous = get_dict_element(data, keys[:-1])

    if previous is not None:
        previous[keys[-1]] = value
        return True

    return False


def add_to_dump(source: PathSource, data, keys: tuple | None = None):
    dump_path = source.get_dump_path()
    if os.path.exists(dump_path):
        with open(dump_path, "rb") as f:
            dump = pickle.load(f)
    else:
        dump = {}

    if keys is None:
        divided_path = dump_path.split("/")
        keys = (divided_path[-2], divided_path[-1].replace(".mp4", ""))

    # Given dump as a nested dictionary, check if every key exists, otherwhise it creates a dictionary at that key level
    for i in range(1, len(keys)):
        sub_list = keys[:i]
        if get_dict_element(dump, sub_list) is None:
            set_dict_element(dump, sub_list, {})
    set_dict_element(dump, keys, data)

    with open(dump_path, "wb") as f:
        pickle.dump(dump, f)


def dump_data(source: PathSource, data, force=False):
    dump_path = source.get_dump_path()

    if force:
        with open(dump_path, "rb") as f:
            data = pickle.load(f) | data

    with open(source.get_dump_path(), "wb") as f:
        pickle.dump(data, f)
