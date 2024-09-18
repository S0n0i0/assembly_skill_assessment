import cv2
import lmdb
import numpy as np
import os


def extract_by_key(env, key, verbose=False):
    """
    input:
        env: lmdb environment initialized (see main function)
        key: the frame number in lmdb key format for which the feature is to be extracted
             the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
             e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84346135_mono10bit/HMC_84346135_mono10bit_0000000001.jpg
    output: a 2048-D np-array (TSM feature corresponding to the key)
    """
    data = env.get(key.strip().encode("utf-8"))
    if verbose and data is None:
        print(f"[ERROR] Key {key} does not exist !!!")
    data = np.frombuffer(data, "float32")  # convert to numpy array
    return data


def clean_lmdb(
    mdb_to_clean: str, mdb_cleaned: str, videos_directory: str, fps_ratio: int, skip_lmdb=[], skip_videos=[]
):

    camera_file_name = mdb_to_clean.split("/")[-1]
    camera_file_name = camera_file_name if camera_file_name != "" else os.getcwd()

    directories = [
        d
        for d in os.listdir(videos_directory)
        if os.path.isdir(os.path.join(videos_directory, d))
    ]

    max_lmdb_frames = {
        directory: int(
            cv2.VideoCapture(
                os.path.join(videos_directory, directory).replace("\\", "/")
                + "/"
                + camera_file_name
                + ".mp4"
            ).get(cv2.CAP_PROP_FRAME_COUNT)
        )
        for i, directory in enumerate(directories)
        if i not in skip_videos
        and os.path.exists(
            os.path.join(videos_directory, directory).replace("\\", "/")
            + "/"
            + camera_file_name
            + ".mp4"
        )
    }
    env_to_clean = lmdb.open(mdb_to_clean, readonly=True, lock=False)
    # get env_to_clean map_size

    try:
        with env_to_clean.begin() as etc:
            video = list(max_lmdb_frames.keys())[0]
            new_key = (
                video
                + "/"
                + camera_file_name
                + "/"
                + camera_file_name
                + "_0000000001.jpg"
            )
            data = extract_by_key(etc, new_key)
            new_size = int(
                np.ceil((data.nbytes / 1e9) * np.sum(list(max_lmdb_frames.values())))
                * 2
            )
            env_cleaned = lmdb.open(
                mdb_cleaned, map_size=new_size * int(1e9), readonly=False, lock=False
            )
            with env_cleaned.begin(write=True) as ec:
                for video, max_frame in max_lmdb_frames.items():
                    for frame in range(1, max_frame):
                        original_frame = frame * fps_ratio
                        base_key = (
                            video
                            + "/"
                            + camera_file_name
                            + "/"
                            + camera_file_name
                            + "_"
                        )
                        original_key = base_key + str(original_frame).zfill(10) + ".jpg"
                        new_key = base_key + str(frame).zfill(10) + ".jpg"
                        data = extract_by_key(etc, original_key)
                        ec.put(new_key.encode(), data.tobytes())

        env_to_clean.close()
        env_cleaned.close()
    except:
        # delete the directory mdb_cleaned and all files within it
        for file in os.listdir(mdb_cleaned):
            os.remove(os.path.join(mdb_cleaned, file))
        os.rmdir(mdb_cleaned)


if __name__ == "__main__":
    directory_to_clean = "C:/old_TSM_features"
    target_directory = "D:/data/TSM_features"
    videos_directory = "D:/data/ego_recordings"
    original_fps = 30
    new_fps = 15
    skip_lmdb = []

    all_directories = os.listdir(directory_to_clean)
    directories = [d for d in all_directories if d not in os.listdir(target_directory) and d not in skip_lmdb]
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
            original_fps // new_fps,
            skip_lmdb,
        )
    print("Done")
