import os
import sys
from tqdm import tqdm
import ffmpeg
import concurrent.futures
import cv2


def process_video(video, sequence, general_view, use_ffmpeg):
    video_path = sequence + "/" + video
    absolute_video_path = os.path.join(origin_paths[general_view], video_path)
    output_directory = os.path.join(target_paths[general_view], sequence, video[:-4])

    output_template = os.path.join(output_directory, video[:-4] + "_frame_%010d.jpg")

    try:
        os.makedirs(output_directory, exist_ok=True)
        if use_ffmpeg:
            (
                ffmpeg.input(absolute_video_path)
                .output(
                    output_template,
                    start_number=1,
                    vcodec="mjpeg",
                    loglevel="quiet",
                )
                .run(quiet=True, overwrite_output=True)
            )
        else:
            cap = cv2.VideoCapture(absolute_video_path)
            idx = 1
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(output_template.format(idx), frame)
                idx += 1
            cap.release()
    except ffmpeg.Error as e:
        print(
            f"Errore durante l'elaborazione del video {video_path}: {e.stderr.decode('utf-8')}",
            file=sys.stderr,
        )


views = ["fixed", "ego"]  # ["ego","fixed"]
origin_paths = {view: f"D:/data/assembly/videos/{view}_recordings/" for view in views}
target_paths = {view: f"D:/data/assembly/{view}_recordings/" for view in views}
image_tmpl = "{:010d}.jpg"
bitrate_target = "8M"
use_ffmpeg = True

sequences = {}
for general_view in views:
    print(f"_Processing {general_view} view")
    sequences[general_view] = [
        d
        for d in os.listdir(origin_paths[general_view])
        if os.path.isdir(os.path.join(origin_paths[general_view], d))
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for sequence in tqdm(
            sequences[general_view], desc=f"{general_view.title()} sequences"
        ):
            output_directory = os.path.join(target_paths[general_view], sequence)
            directory_frames = set()
            os.makedirs(output_directory, exist_ok=True)
            directory_frames.update([f for f in os.listdir(output_directory)])

            videos = [
                f
                for f in os.listdir(os.path.join(origin_paths[general_view], sequence))
                if f.endswith(".mp4")
            ]

            frames = set()
            for video in videos:
                video_path = sequence + "/" + video
                absolute_video_path = os.path.join(
                    origin_paths[general_view], video_path
                )
                if use_ffmpeg:
                    probe = ffmpeg.probe(absolute_video_path)
                    total_frames = int(
                        next(
                            (s for s in probe["streams"] if s["codec_type"] == "video"),
                            {},
                        ).get("nb_frames", 0)
                    )
                else:
                    cap = cv2.VideoCapture(absolute_video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                frames.update(
                    [
                        f"{video[:-4]}_{image_tmpl.format(idx)}"
                        for idx in range(1, total_frames + 1)
                    ]
                )

            missing_frames = frames.difference(directory_frames)

            if len(missing_frames) == 0:
                continue

            for video in videos:
                futures.append(
                    executor.submit(
                        process_video, video, sequence, general_view, use_ffmpeg
                    )
                )

            concurrent.futures.wait(futures)
