import os
import ffmpeg

view = "ego"  # "ego" or "fixed"
sequences_directory = f"D:/data/{view}_recordings/"
sequence_fps = 15
assembly_directory = f"D:/data/assembly/{view}_recordings/"
offsets_file = "D:/data/annotations/coarse-annotations/{view}_offsets.csv"
not_used_files_path = f"D:/data/annotations/coarse-annotations/not_used_videos.csv"
new_not_used = False
not_used_actions_path = (
    f"D:/data/assembly/annotations/action_anticipation/not_used_actions.csv"
)
coarse_directory = "D:/data/annotations/coarse-annotations/coarse_labels/"
fine_directory = "D:/data/annotations/action_anticipation/"
assembly_fine_directory = "D:/data/assembly/annotations/action_anticipation/"
annotations_fps = 30
include_non_cropped = False

splits = ["train", "validation", "validation_challenge", "test", "test_challenge"]
trainval_first_line = [
    "id",
    "video",
    "start_frame",
    "end_frame",
    "action_id",
    "verb_id",
    "noun_id",
    "action_cls",
    "verb_cls",
    "noun_cls",
    "toy_id",
    "toy_name",
    "is_shared",
    "is_rgb",
]
first_lines = {
    "train": trainval_first_line,
    "validation": trainval_first_line,
    "validation_challenge": trainval_first_line,
    "test": [
        "id",
        "video",
        "start_frame",
        "end_frame",
        "is_shared",
        "is_rgb",
    ],
    "test_challenge": [
        "id",
        "video",
        "start_frame",
        "end_frame",
        "is_shared",
        "is_rgb",
    ],
    "actions": [
        "id",
        "action_id",
        "verb_id",
        "noun_id",
        "action_cls",
        "verb_cls",
        "noun_cls",
    ],
}

modes = [
    "remove_disassembly_rows",
]  # ["crop_assembly", "remove_disassembly_rows"]

first_frames = {}
if "crop_assembly" in modes:
    first_frames = {
        d: {}
        for d in os.listdir(sequences_directory)
        if os.path.isdir(os.path.join(sequences_directory, d))
    }
    not_cropped = {}

    for ff in first_frames:
        videos_directory = os.path.join(sequences_directory, ff)
        assembly_videos_directory = os.path.join(assembly_directory, ff)
        assembly_frames_file_path = os.path.join(
            coarse_directory, "assembly_" + ff + ".txt"
        )

        exists_coarse = os.path.exists(assembly_frames_file_path)
        if exists_coarse or include_non_cropped:
            first_frames[ff] = {
                v: 0
                for v in os.listdir(videos_directory)
                if os.path.isfile(os.path.join(videos_directory, v))
            }
        if not exists_coarse:
            not_cropped[ff] = "no_coarse_annotations"
            continue
        if len(first_frames[ff]) == 0:
            not_cropped[ff] = "no_ego/fixed_videos"
            continue

        if not os.path.exists(assembly_videos_directory):
            os.makedirs(assembly_videos_directory)
        with open(assembly_frames_file_path) as assembly_frames_file:
            # map start_frame, which is at annotations_fps, to the corresponding frame in the video, which is at sequence_fps
            start_frame = int(assembly_frames_file.readline().split("	")[0])
        start_second = start_frame // annotations_fps

        for v in first_frames[ff]:
            first_frames[ff][v] = start_frame
            video_path = os.path.join(videos_directory, v)
            assembly_video_path = os.path.join(assembly_videos_directory, v)

            if os.path.exists(assembly_video_path):
                continue

            # crop the video to the assembly
            ffmpeg.input(video_path, ss=start_second).output(
                assembly_video_path, vcodec="h264_nvenc", loglevel="quiet"
            ).run(
                # overwrite_output=True
            )
        if len(first_frames[ff]) > 0:
            print(
                f"Cropped {ff} at frame {first_frames[ff][list(first_frames[ff].keys())[0]]}"
            )
        else:
            print(f"No videos found for {ff}")

    if not new_not_used:
        with open(not_used_files_path, "r") as f:
            lines = set([l.split(",")[0] for l in f.readlines()])
    with open(not_used_files_path, "w" if new_not_used else "a") as o:
        for name in not_cropped:
            if new_not_used or name not in lines:
                o.write(f"{name},{not_cropped[name]}\n")

    # Save first frames in offsets_file with the format: directory_name/video_name,start_frame
    with open(offsets_file.format(view=view), "w") as offsets:
        offsets.write("id,video,start_frame\n")
        count = 0
        for ff in first_frames:
            if len(first_frames[ff]) == 0:
                continue
            for v in first_frames[ff]:
                offsets.write(f"{count},{ff + '/' + v},{str(first_frames[ff][v])}\n")
            count += 1

if "remove_disassembly_rows" in modes:
    if len(first_frames) == 0:
        with open(offsets_file.format(view="ego"), "r") as offsets:
            lines = offsets.readlines()
        for line in lines[1:]:
            id, video, start_frame = line.split(",")
            name, view = video.split("/")
            if name not in first_frames:
                first_frames[name] = {}
            first_frames[name][view] = int(start_frame)

    assembly_actions = set()
    for split in splits:
        print(f"Processing {split}")
        with open(os.path.join(fine_directory, split + ".csv")) as f:
            lines = f.readlines()

        count = 0
        with open(os.path.join(assembly_fine_directory, split + ".csv"), "w") as f:
            f.write(",".join(first_lines[split]) + "\n")
            count = -1
            last_id = -1
            for line in lines[1:]:
                line = line.strip().split(",")
                id = int(line[0])
                start_frame = int(line[2])

                if "_challenge" not in split:
                    name, view = line[1].split("/")
                else:
                    name = line[1]
                    view = (
                        list(first_frames[name].keys())[0]
                        if name in first_frames
                        else None
                    )

                if split != "test" and split != "test_challenge":
                    action_id = int(line[4])
                    if (
                        name not in first_frames
                        or start_frame < first_frames[name][view]
                    ):
                        continue
                    assembly_actions.add(action_id)
                else:

                    if (
                        name not in first_frames
                        or start_frame < first_frames[name][view]
                    ):
                        continue

                if id != last_id:
                    count += 1
                    last_id = id
                f.write(f"{count},{','.join(line[1:])}\n")

    print("\nProcessing actions")
    with open(os.path.join(fine_directory, "actions.csv")) as f:
        lines = f.readlines()
        actions = set([int(l.split(",")[1]) for l in lines[1:]])

    with open(os.path.join(assembly_fine_directory, "actions.csv"), "w") as f:
        with open(not_used_actions_path, "w") as o:
            f.write(",".join(first_lines["actions"]) + "\n")
            o.write(",".join(first_lines["actions"][1:]) + "\n")
            count = 0
            actions_to_remove = list(actions - assembly_actions)
            actions_to_remove.sort()
            print(
                f"Actions to remove ({len(actions)} -> {len(assembly_actions)}): {len(actions_to_remove)}"
            )
            print(actions_to_remove)
            for line in lines[1:]:
                line = line.strip().split(",")
                if int(line[1]) not in actions_to_remove:
                    f.write(f"{count},{','.join(line[1:])}\n")
                    count += 1
                else:
                    o.write(f"{','.join(line[1:])}\n")
