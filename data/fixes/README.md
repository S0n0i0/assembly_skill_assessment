# Program combinations

## Data download

1) Download annotations from drive

2) Execute one command from the following to download videos:
    ```sh
    py ./data/fixes/decrease_and_move.py
    py ./data/fixes/decrease.py
    py ./data/fixes/download_from_remote.py
    py ./data/fixes/recording_downloader.py
    ```

3) Download TSM_features

## Data cleaning

1) If necessary remove fixed views:
    ```sh
    py ./data/fixes/remove_fixed_views.py
    ```

2) Extract assembly data, clean not used video and actions and map them
    ```sh
    py ./data/fixes/extract_assembly.py
    ```

3) Gen all the needed splits executing all the following commands:
    ```sh
    py ./data/fixes/gen_fine_skills_splits.py
    py ./data/fixes/gen_fine_labels.py
    py ./data/fixes/gen_coarse_splits.py
    py ./data/fixes/gen_grouped_skill_splits.py
    ```

4) Extract frames from videos
    ```sh
    py ./data/fixes/frame_extractor.py
    ```

5) Remove useless data from TSM_features
    ```sh
    py ./data/fixes/remove_lmdb_frames.py
    ```