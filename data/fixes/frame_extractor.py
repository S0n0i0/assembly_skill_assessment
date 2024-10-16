import os
from tqdm import tqdm
import cv2

views = ["fixed", "ego"]  # ["ego","fixed"]
origin_paths = {view: f"D:/data/assembly/{view}_recordings/" for view in views}
target_paths = {view: f"C:/tempa/{view}_recordings/" for view in views}
offsets_paths = {
    view: f"D:/data/assembly/annotations/{view}_offsets.csv" for view in views
}
image_tmpl = "frame_{:010d}.jpg"
compression_percentage = 85

sequences = {}
for general_view in views:
    print(f"_Processing {general_view} view")
    sequences[general_view] = [
        d
        for d in os.listdir(origin_paths[general_view])
        if os.path.isdir(os.path.join(origin_paths[general_view], d))
    ]

    with open(offsets_paths[general_view], "r") as f:
        offsets = {
            line.strip().split(",")[1]: int(line.strip().split(",")[2])
            for line in f.readlines()[1:]
        }

    for sequence in tqdm(
        sequences[general_view], desc=f"{general_view.title()} sequences"
    ):
        output_directory = os.path.join(target_paths[general_view], sequence)
        directory_frames = set()
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        else:
            directory_frames.update([f for f in os.listdir(output_directory)])
        videos = [
            f
            for f in os.listdir(os.path.join(origin_paths[general_view], sequence))
            if f.endswith(".mp4")
        ]

        frames = set()
        for video in videos:
            video_path = sequence + "/" + video
            absolute_video_path = os.path.join(origin_paths[general_view], video_path)
            cap = cv2.VideoCapture(absolute_video_path)
            if not cap.isOpened():
                print(f"{video_path}: Impossibile aprire il video.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames.update(
                    [
                        f"{video[:-4]}_{image_tmpl.format(idx)}"  # offsets[video_path] +
                        for idx in range(total_frames)
                    ]
                )
        missing_frames = frames.difference(directory_frames)

        if len(missing_frames) == 0:
            continue

        for i, video in enumerate(videos):
            video_path = sequence + "/" + video
            absolute_video_path = os.path.join(origin_paths[general_view], video_path)

            # open video and save frames in output_directory
            cap = cv2.VideoCapture(absolute_video_path)
            idx = 1  # offsets[video_path]
            if not cap.isOpened():
                print(f"{video_path}: Impossibile aprire il video.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for _ in tqdm(
                    range(total_frames), desc=f"{i+1}/{len(videos)}", leave=False
                ):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"{video_path}: Impossibile leggere un frame.")
                        break

                    image_path = os.path.join(
                        output_directory,
                        f"{video[:-4]}_{image_tmpl.format(idx)}",
                    )
                    # Check if the image already exists
                    if os.path.exists(image_path):
                        continue

                    cv2.imwrite(
                        image_path,
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, compression_percentage],
                    )
                    idx += 1
