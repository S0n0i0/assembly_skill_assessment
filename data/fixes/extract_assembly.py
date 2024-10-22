import os
import ffmpeg
from enum import Enum
import shutil
import cv2
from tqdm import tqdm

from utils.constants import first_lines


class not_used_codes(Enum):
    NO_COARSE = "no_coarse_annotations"
    NO_VIDEOS = "no_ego/fixed_videos"
    NO_ASSEMBLY = "no_ego/fixed_assembly"
    NO_ASSEMBLY_VIDEO = "no_assembly"


def get_contrary_view(view):
    return "ego" if view == "fixed" else "fixed"


views = ["ego", "fixed"]  # ["ego","fixed"]
annotations_types = {
    "fine": lambda x: int(x.split(",")[1]),
    "coarse": lambda x: x.split(",")[4],
}
sequences_paths = {view: f"D:/data/{view}_recordings/" for view in views}
sequence_fps = 15
assembly_directories = {
    view: f"D:/data/assembly/videos/{view}_recordings/" for view in views
}
offsets_paths = {
    view: f"D:/data/assembly/annotations/{view}_offsets.csv" for view in views
}
not_used_files_path = "D:/data/assembly/annotations/not_used_videos.csv"
not_used_actions_path = {
    "fine": "D:/data/assembly/annotations/action_anticipation/not_used_actions.csv",
    "coarse": "D:/data/assembly/annotations/coarse-annotations/not_used_actions.csv",
}
annotations_directories = {
    "coarse": "D:/data/annotations/coarse-annotations/",
    "fine": "D:/data/annotations/action_anticipation/",
    "skills": "D:/data/annotations/skill_labels/",
}
assembly_annotations_directories = {
    "coarse": "D:/data/assembly/annotations/coarse-annotations/",
    "fine": "D:/data/assembly/annotations/action_anticipation/cropped/",
    "skills": "D:/data/assembly/annotations/skill_labels/",
}
annotations_fps = 30
include_non_cropped = False

splits = ["train", "validation", "validation_challenge", "test", "test_challenge"]

modes = [
    "crop_assembly",
    "check_views_diffs",
    "remove_disassembly_rows",
]  # ["crop_assembly", "check_views_diffs", "remove_disassembly_rows"]
remove_disassembly_rows = {
    "not_used": True,
    "splits_actions": True,
}

first_frames = {view: {} for view in views}
not_used = {}
if "crop_assembly" in modes:
    print("Cropping assembly videos")
    for general_view in views:
        print(f"_Processing {general_view} view")
        first_frames[general_view] = {
            d: {}
            for d in os.listdir(sequences_paths[general_view])
            if os.path.isdir(os.path.join(sequences_paths[general_view], d))
        }

        for sequence in tqdm(first_frames[general_view]):
            if sequence in not_used:
                continue

            videos_directory = os.path.join(sequences_paths[general_view], sequence)
            assembly_videos_directory = os.path.join(
                assembly_directories[general_view], sequence
            )
            assembly_frames_file_path = os.path.join(
                annotations_directories["coarse"],
                "coarse_labels/assembly_" + sequence + ".txt",
            )

            exists_coarse = os.path.exists(assembly_frames_file_path)
            if exists_coarse or include_non_cropped:
                first_frames[general_view][sequence] = {
                    v: -1
                    for v in os.listdir(videos_directory)
                    if os.path.isfile(os.path.join(videos_directory, v))
                }
            if not exists_coarse:
                not_used[sequence] = not_used_codes.NO_COARSE.value
                continue
            if len(first_frames[general_view][sequence]) == 0:
                not_used[sequence] = not_used_codes.NO_VIDEOS.value
                continue

            if not os.path.exists(assembly_videos_directory):
                os.makedirs(assembly_videos_directory)
            with open(assembly_frames_file_path) as assembly_frames_file:
                # map start_frame, which is at annotations_fps, to the corresponding frame in the video, which is at sequence_fps
                start_frame = int(assembly_frames_file.readline().split("	")[0])
            start_second = start_frame / annotations_fps

            for v in first_frames[general_view][sequence]:
                video_path = os.path.join(videos_directory, v)
                assembly_video_path = os.path.join(assembly_videos_directory, v)
                video = cv2.VideoCapture(video_path)

                if start_second > video.get(cv2.CAP_PROP_FRAME_COUNT) / sequence_fps:
                    continue

                first_frames[general_view][sequence][v] = start_frame

                if os.path.exists(assembly_video_path):
                    continue

                # crop the video to the assembly
                ffmpeg.input(video_path, ss=start_second).output(
                    assembly_video_path, vcodec="h264_nvenc", loglevel="quiet"
                ).run(
                    # overwrite_output=True
                )

            no_assembly = [
                offset == -1 for offset in first_frames[general_view][sequence].values()
            ]
            if all(no_assembly):
                not_used[sequence] = not_used_codes.NO_ASSEMBLY.value
            elif any(no_assembly):
                for view, offset in first_frames[general_view][sequence].items():
                    if offset == -1:
                        not_used[sequence + "/" + view] = (
                            not_used_codes.NO_ASSEMBLY_VIDEO.value
                        )

    for general_view in views:
        with open(offsets_paths[general_view], "w") as offsets:
            offsets.write(",".join(first_lines["offsets"]) + "\n")
            i = 0
            first_frames_keys = list(first_frames[general_view].keys())
            while i < len(first_frames_keys):
                sequence = first_frames_keys[i]
                if sequence in not_used:
                    del first_frames[general_view][sequence]
                    first_frames_keys.pop(i)
                    continue
                j = 0
                offsets_keys = list(first_frames[general_view][sequence].keys())
                while j < len(offsets_keys):
                    v = offsets_keys[j]
                    if sequence + "/" + v in not_used:
                        del first_frames[general_view][sequence][v]
                        offsets_keys.pop(j)
                        continue
                    offsets.write(
                        f"{i},{sequence + '/' + v},{str(first_frames[general_view][sequence][v])}\n"
                    )
                    j += 1
                i += 1

sequences_to_remove = {view: set() for view in views}
non_common = {}
sequences = {view: set(os.listdir(assembly_directories[view])) for view in views}
all_sequences = set.intersection(*sequences.values())
used = all_sequences.copy()
if "check_views_diffs" in modes:
    print("\nChecking views differences")

    # Find sequences that are in one but not the other
    print("Sequences:")
    for general_view in views:
        print(f"- {general_view} sequences: {len(sequences[general_view])}")

    print("Non common sequences:")
    for vs in zip(views, views[::-1]):
        non_common[vs[0]] = sequences[vs[0]] - sequences[vs[1]]
        print(
            f"- {vs[0]} sequences but not in {vs[1]} ({len(non_common[vs[0]])}):",
            non_common[vs[0]],
        )

    for general_view in views:
        for name in non_common[general_view]:
            if name in not_used:
                continue
            not_used[name] = not_used_codes.NO_VIDEOS.value
        sequences_to_remove[general_view] = non_common[general_view]
    used -= set(not_used.keys())

if remove_disassembly_rows["not_used"]:
    print("\nRemoving sequences that are not used")
    print("_Processing fine grained splits")
    for general_view in views:
        for name in sequences_to_remove[general_view]:
            for file in os.listdir(
                os.path.join(assembly_directories[general_view], name)
            ):
                os.remove(os.path.join(assembly_directories[general_view], name, file))
            os.rmdir(os.path.join(assembly_directories[general_view], name))

    print("_Processing coarse labels")
    count = 0
    disassembly_count = 0
    directories = os.listdir(
        os.path.join(annotations_directories["coarse"], "coarse_labels")
    )
    for coarse_sequence in directories:
        if "_".join(coarse_sequence.split("_")[1]) in sequences_to_remove["ego"]:
            count += 1
            continue
        elif "disassembly_" in coarse_sequence:
            disassembly_count += 1
            continue
        elif os.path.exists(
            os.path.join(
                assembly_annotations_directories["coarse"],
                "coarse_labels/",
                coarse_sequence,
            )
        ):
            continue

        shutil.copy(
            os.path.join(
                annotations_directories["coarse"], "coarse_labels/", coarse_sequence
            ),
            os.path.join(assembly_annotations_directories["coarse"], "coarse_labels"),
        )
    print(
        f"- Removed sequences ({len(directories) - disassembly_count} -> {len(directories) - disassembly_count - count}): {count}"
    )

    print("_Processing skills labels")
    for skill_level_file in os.listdir(annotations_directories["skills"]):
        count = 0
        with open(
            os.path.join(annotations_directories["skills"], skill_level_file), "r"
        ) as f:
            lines = f.readlines()

        with open(
            os.path.join(assembly_annotations_directories["skills"], skill_level_file),
            "w",
        ) as f:
            for line in lines:
                line = line.strip()
                if line in used:
                    f.write(line + "\n")
                else:
                    count += 1
            print(
                f"- Removed sequences of level {skill_level_file[:-4].split('_')[1]} ({len(lines)} -> {len(lines) - count}): {count}"
            )

    print("_Listing not used videos")
    with open(not_used_files_path, "w") as o:
        for name in not_used:
            o.write(f"{name},{not_used[name]}\n")

    print("_Writing down offsets")
    for general_view in views:
        with open(offsets_paths[general_view], "r") as f:
            lines = f.readlines()
        with open(offsets_paths[general_view], "w") as f:
            f.write(",".join(first_lines["offsets"]) + "\n")
            count = -1
            last_id = -1
            for line in lines[1:]:
                line = line.split(",")
                id = int(line[0])
                name = line[1].split("/")[0]
                if name not in sequences_to_remove[general_view]:
                    if id != last_id:
                        count += 1
                        last_id = id
                    f.write(f"{count},{','.join(line[1:])}")

if remove_disassembly_rows["splits_actions"]:
    print("\nRemoving disassembly rows from splits")
    if len(first_frames["ego"]) == 0:
        with open(offsets_paths["ego"], "r") as offsets:
            lines = offsets.readlines()
        for line in lines[1:]:
            id, video, start_frame = line.split(",")
            name, general_view = video.split("/")
            if name not in first_frames["ego"]:
                first_frames["ego"][name] = {}
            first_frames["ego"][name][general_view] = int(start_frame)

    assembly_actions = {annotation_type: set() for annotation_type in annotations_types}
    for split in splits:
        print(f"Processing {split}")
        with open(os.path.join(annotations_directories["fine"], split + ".csv")) as f:
            lines = f.readlines()

        count = 0
        with open(
            os.path.join(assembly_annotations_directories["fine"], split + ".csv"), "w"
        ) as f:
            f.write(",".join(first_lines["splits"][split]) + "\n")
            count = -1
            last_id = -1
            for line in lines[1:]:
                line = line.strip().split(",")
                id = int(line[0])
                start_frame = int(line[2])

                if "_challenge" not in split:
                    name, general_view = line[1].split("/")
                else:
                    name = line[1]
                    general_view = (
                        (
                            list(first_frames["ego"][name].keys())[0]
                            if name in first_frames["ego"]
                            else None
                        )
                        if name in first_frames["ego"]
                        and len(first_frames["ego"][name]) > 0
                        else None
                    )

                if (
                    name not in first_frames["ego"]
                    or general_view not in first_frames["ego"][name]
                    or start_frame < first_frames["ego"][name][general_view]
                ):
                    continue

                if split != "test" and split != "test_challenge":
                    action_id = int(line[4])
                    assembly_actions["fine"].add(action_id)

                if id != last_id:
                    count += 1
                    last_id = id
                    with open(
                        os.path.join(
                            assembly_annotations_directories["coarse"],
                            "coarse_labels/assembly_" + name + ".txt",
                        ),
                        "r",
                    ) as o:
                        assembly_actions["coarse"].update(
                            [line.split("	")[2] for line in o.readlines()]
                        )

                f.write(f"{count},{','.join(line[1:])}\n")

    print("\nProcessing actions")
    for annotation_type in annotations_types:
        with open(
            os.path.join(annotations_directories[annotation_type], "actions.csv"), "r"
        ) as f:
            lines = f.readlines()
            actions = set([annotations_types[annotation_type](l) for l in lines[1:]])

        with open(
            os.path.join(
                assembly_annotations_directories[annotation_type], "actions.csv"
            ),
            "w",
        ) as f, open(not_used_actions_path[annotation_type], "w") as o:
            f.write(",".join(first_lines["actions"]) + "\n")
            o.write(
                ",".join(
                    first_lines["actions"][
                        (1 if first_lines["actions"] == "id" else 0) :
                    ]
                )
                + "\n"
            )
            count = 0
            actions_to_remove = list(actions - assembly_actions[annotation_type])
            actions_to_remove.sort()
            print(
                f"{annotation_type.title()} actions to remove ({len(actions)} -> {len(assembly_actions[annotation_type])}): {len(actions_to_remove)}"
            )
            print(actions_to_remove)
            for line in lines[1:]:
                divided_line = line.strip().split(",")
                if annotations_types[annotation_type](line) not in actions_to_remove:
                    f.write(f"{count},{','.join(divided_line[1:])}\n")
                    count += 1
                else:
                    o.write(f"{','.join(divided_line[1:])}\n")
