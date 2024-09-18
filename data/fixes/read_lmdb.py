"""
    helper functions to read the TSM_feature lmdb
    run this with a command line argument describing the path to the lmdb
    e.g. python read_lmdb.py TSM_features/C10095_rgb 
"""

import os
import sys
import lmdb
import numpy as np
from zipfile import ZipFile
import cv2


# path to the lmdb file you want to read as a command line argument
# lmdb_path = sys.argv[1]


# iterate over the entire lmdb and output all files
def extract_all_features(env):
    """
    input:
        env: lmdb environment loaded (see main function)
    output: a dictionary with key as the path_to_frame and value as the TSM feature (2048-D np-array)
            the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
            e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg
    """
    # ALL THE FRAME NUMBERS ARE AT 30FPS !!!

    all_features = set()

    print("Iterating over the entire lmdb. This may take some time...")
    with env.begin() as e:
        cursor = e.cursor()

        videos = {}
        for file, data in cursor:
            frame = file.decode("utf-8")
            data = np.frombuffer(data, dtype=np.float32)
            if data.shape[0] == 2048:
                v = frame.split("/")[0]
                videos[v] = 0 if v not in videos else videos[v] + 1
                # all_features.add(frame)
            else:
                print(frame, data.shape)

    print(f"Features for {len(all_features)} frames loaded.")
    return all_features, videos


# extract the feature for a particular key
def extract_by_key(env, key):
    """
    input:
        env: lmdb environment loaded (see main function)
        key: the frame number in lmdb key format for which the feature is to be extracted
             the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
             e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg
    output: a 2048-D np-array (TSM feature corresponding to the key)
    """
    with env.begin() as e:
        # get available keys
        cursor = e.cursor()

        # Otteniamo la prima chiave
        if cursor.first():
            k, value = cursor.item()
            print(f"La prima chiave è: {k.decode('utf-8')}")
        else:
            print("Il database è vuoto.")
        data = e.get(key.strip().encode("utf-8"))
        if data is None:
            print(f"[ERROR] Key {key} does not exist !!!")
            exit()
        data = np.frombuffer(data, "float32")  # convert to numpy array
    return data


# main function
if __name__ == "__main__":
    camera_dir = "HMC_84346135_mono10bit"
    # load the lmdb environment from the path
    lmdb_path = "D:/data/TSM_features/" + camera_dir

    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    mode = 1
    if mode == 0:
        # extract_by_key() example
        # key = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84346135_mono10bit/HMC_84346135_mono10bit_0000000001.jpg"
        # key = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84346135_mono10bit/HMC_84346135_mono10bit_0000016712.jpg"
        # key = "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84346135_mono10bit/HMC_84346135_mono10bit_0000008356.jpg"
        # 0.0078125 MB
        base_key = (
            "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/"
            + camera_dir
        )
        video_path = "D:/data/ego_recordings/" + base_key + ".mp4"
        # get the max frame number
        cap = cv2.VideoCapture(video_path)
        print("Ciao")
        max_frame = int(np.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
        key = base_key + f"/{camera_dir}_{str(1).zfill(10)}.jpg"
        data = extract_by_key(env, key)
        print(1, data.shape)
        key = base_key + f"/{camera_dir}_{str(2).zfill(10)}.jpg"
        data = extract_by_key(env, key)
        print(2, data.shape)
        # add max_frame with 10 digits to key
        key = base_key + f"/{camera_dir}_{str(max_frame).zfill(10)}.jpg"
        data = extract_by_key(env, key)
        print(max_frame, data.shape)
        # get GB size of the data
        print(f"Size of the data: {data.nbytes / 1e9} GB")
        print("This video last " + str(max_frame / 30 / 60) + " minutes.")
    elif mode == 1:
        _, v = extract_all_features(env)
        print(len(list(v.keys())), np.sum(list(v.values())))
