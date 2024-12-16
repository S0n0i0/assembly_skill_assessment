import cv2
import lmdb
import numpy as np
import os
from tqdm import tqdm

from utils.functions import extract_by_key


def clean_lmdb(
    mdb_to_clean: str,
    mdb_cleaned: str,
    videos_directory: str,
    original_fps: float,
    new_fps: float,
    not_moved_frames_file: str,
    offsets_file: str | None = None,
    offsets_fps: float | None = None,
    skip_lmdb=[],
    skip_videos=[],
):

    camera_file_name = mdb_to_clean.split("/")[-1]
    camera_file_name = camera_file_name if camera_file_name != "" else os.getcwd()

    directories = [
        d
        for d in os.listdir(videos_directory)
        if os.path.isdir(os.path.join(videos_directory, d))
    ]

    if offsets_file is not None:
        with open(offsets_file, "r") as f:
            offsets = {
                line.split(",")[1].split("/")[0]: int(line.strip().split(",")[2])
                for line in f.readlines()[1:]
                if line.split(",")[1].split("/")[1][:-4] == camera_file_name
            }

    max_lmdb_frames = {
        sequence: int(
            cv2.VideoCapture(
                os.path.join(videos_directory, sequence).replace("\\", "/")
                + "/"
                + camera_file_name
                + ".mp4"
            ).get(cv2.CAP_PROP_FRAME_COUNT)
        )
        for i, sequence in enumerate(directories)
        if i not in skip_videos
        and os.path.exists(
            os.path.join(videos_directory, sequence).replace("\\", "/")
            + "/"
            + camera_file_name
            + ".mp4"
        )
    }
    env_to_clean = lmdb.open(mdb_to_clean, readonly=True, lock=False)
    # get env_to_clean map_size

    try:
        with env_to_clean.begin() as etc:
            sequence = list(max_lmdb_frames.keys())[0]
            new_key = (
                sequence
                + "/"
                + camera_file_name
                + "/"
                + camera_file_name
                + "_0000000001.jpg"
            )
            data = extract_by_key(etc, new_key)
            new_size = int(
                np.ceil((data.nbytes / 1e9) * np.sum(sum(max_lmdb_frames.values()))) * 2
            )
            env_cleaned = lmdb.open(
                mdb_cleaned, map_size=new_size * int(1e9), readonly=False, lock=False
            )
            with env_cleaned.begin(write=True) as ec:
                with open(not_moved_frames_file, "a") as not_moved_frames:
                    for sequence, max_frame in tqdm(max_lmdb_frames.items()):
                        for frame_index in range(1, max_frame):
                            original_frame_index = int(
                                (
                                    frame_index / new_fps
                                    + offsets[sequence] / offsets_fps
                                )
                                * original_fps
                            )
                            base_key = (
                                sequence
                                + "/"
                                + camera_file_name
                                + "/"
                                + camera_file_name
                                + "_"
                            )
                            # nusar-2021_action_both_9013-a02_9013_user_id_2021-02-02_130807/HMC_21110305_mono10bit/HMC_21110305_mono10bit_0000008645.jpg
                            # nusar-2021_action_both_9013-a02_9013_user_id_2021-02-02_130807/HMC_21110305_mono10bit/HMC_21110305_mono10bit_0000021254.jpg
                            original_key = (
                                base_key + str(original_frame_index).zfill(10) + ".jpg"
                            )
                            new_key = base_key + str(frame_index).zfill(10) + ".jpg"
                            data = extract_by_key(etc, original_key)
                            if data is None:
                                not_moved_frames.write(original_key + "\n")
                                continue
                                # raise Exception("Key not found: " + original_key)
                            ec.put(new_key.encode(), data.tobytes())

        env_to_clean.close()
        env_cleaned.close()
        return True
    except Exception as e:
        print(e)
        # delete the directory mdb_cleaned and all files within it
        env_to_clean.close()
        env_cleaned.close()
        for file in os.listdir(mdb_cleaned):
            os.remove(os.path.join(mdb_cleaned, file))
        os.rmdir(mdb_cleaned)
        return False


if __name__ == "__main__":
    directory_to_clean = "C:/tempa/TSM_features"
    target_directory = "D:/data/TSM_features"
    videos_directory = "D:/data/ego_recordings"
    not_moved_frames_file = "D:/data/TSM_features/not_moved.csv"
    offsets_file = "D:/data/annotations/ego_offsets.csv"
    original_fps = 30
    new_fps = 15
    offsets_fps = 30
    skip_lmdb = [
        # "HMC_21110305_mono10bit",
        # "HMC_21176623_mono10bit",
        # "HMC_21176875_mono10bit",
        # "HMC_21179183_mono10bit",
        # "HMC_84346135_mono10bit",
        # "HMC_84347414_mono10bit",
        # "HMC_84355350_mono10bit",
        # "HMC_84358933_mono10bit",
    ]

    if os.path.exists(not_moved_frames_file):
        os.remove(not_moved_frames_file)
    all_directories = os.listdir(directory_to_clean)
    directories = [
        d
        for d in all_directories
        if d not in os.listdir(target_directory) and d not in skip_lmdb
    ]
    for d in all_directories:
        if d not in directories:
            if d in skip_lmdb:
                print(f"Skipping {d}")
            else:
                print(f"TSM_features {d} already cleaned")
            continue
        print(f"Cleaning TSM_features: {d}")
        clean_lmdb(
            os.path.join(directory_to_clean, d).replace("\\", "/"),
            os.path.join(target_directory, d).replace("\\", "/"),
            videos_directory,
            original_fps,
            new_fps,
            not_moved_frames_file,
            offsets_file,
            offsets_fps,
            skip_lmdb,
        )
    print("Done")
