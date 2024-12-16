import os
import ffmpeg
from enum import Enum
import shutil
import cv2
from tqdm import tqdm
import sys

from utils.constants import first_lines
from utils.functions import get_view_frames_data


class not_used_codes(Enum):
    NO_SKILL = "no_skill_level"
    NO_COARSE = "no_coarse_annotations"
    NO_VIDEOS = "no_ego/fixed_videos"
    NO_ASSEMBLY = "no_ego/fixed_assembly"
    NO_ASSEMBLY_VIDEO = "no_assembly"
    VIDEOS_TOO_SHORT = "ego/fixed_videos_too_short"


def get_contrary_view(view):
    return "ego" if view == "fixed" else "fixed"


general_views = ["ego", "fixed"]  # ["ego","fixed"]
action_getter = {
    "fine": {
        "id": lambda x, is_list=False: int(x[1] if is_list else x.split(",")[1]),
        "cls": lambda x, is_list=False: x[4] if is_list else x.split(",")[4],
    },
    "coarse": {
        "id": lambda x, is_list=False: int(x[1] if is_list else x.split(",")[1]),
        "cls": lambda x, is_list=False: x[4] if is_list else x.split(",")[4],
    },
}
edit_all_if_too_short = {
    "ego": False,
    "fixed": True,
}
sequences_paths = {view: f"D:/data/old/{view}_recordings/" for view in general_views}
sequence_fps = 15
assembly_directories = {
    view: f"D:/data/videos/{view}_recordings/" for view in general_views
}
offsets_paths = {
    view: f"D:/data/annotations/{view}_offsets.csv" for view in general_views
}
not_used_files_path = "D:/data/annotations/not_used_videos.csv"
not_used_actions_path = {
    "fine": "D:/data/annotations/action_anticipation/not_used_actions.csv",
    "coarse": "D:/data/annotations/coarse-annotations/not_used_actions.csv",
}
actions_mapping_path = {
    "fine": "D:/data/annotations/action_anticipation/actions_mapping.csv",
    "coarse": "D:/data/annotations/coarse-annotations/actions_mapping.csv",
}
annotations_directories = {
    "coarse": "D:/data/old/annotations/coarse-annotations/",
    "fine": "D:/data/old/annotations/action_anticipation/",
    "skills": "D:/data/old/annotations/skill_labels/",
}
assembly_annotations_directories = {
    "coarse": "D:/data/annotations/coarse-annotations/",
    "fine": "D:/data/annotations/action_anticipation/cropped/",
    "skills": "D:/data/annotations/skill_evaluation/skill_labels/",
}
annotations_fps = 30

splits = ["train", "validation", "validation_challenge", "test", "test_challenge"]

extract_assembly = True
remove_disassembly_rows = {
    "labels": True,
    "splits_actions": True,
    "map_actions": True,  # Also "splits_actions" must be True
}

frames_data = {view: {} for view in general_views}
not_used = {}
if extract_assembly:
    skill_sequences = set()
    for skill_level_file in os.listdir(annotations_directories["skills"]):
        with open(
            os.path.join(annotations_directories["skills"], skill_level_file)
        ) as f:
            for line in f.readlines():
                skill_sequences.add(line.strip())

    shorten_propagation = (
        {}
    )  # Contain tuples (new_end_frame, list[<general_view already propagated>]) of sequences who need the propagation of new_end_frame
    print("Analyzing videos")
    for general_view in general_views:
        print(f"_Processing {general_view} view")
        frames_data[general_view] = {
            d: {}
            for d in os.listdir(sequences_paths[general_view])
            if os.path.isdir(os.path.join(sequences_paths[general_view], d))
        }

        for sequence in tqdm(frames_data[general_view]):
            if sequence in not_used:
                continue
            propagate_shorten = sequence in shorten_propagation

            videos_directory = os.path.join(sequences_paths[general_view], sequence)
            assembly_frames_file_path = os.path.join(
                annotations_directories["coarse"],
                "coarse_labels/assembly_" + sequence + ".txt",
            )

            exists_skill = sequence in skill_sequences
            if not exists_skill:
                not_used[sequence] = not_used_codes.NO_SKILL.value
                continue

            exists_coarse = os.path.exists(assembly_frames_file_path)
            if exists_coarse:
                frames_data[general_view][sequence] = {
                    v: {
                        "first_frame": -1,
                        "new_end_frame": (
                            shorten_propagation[sequence][0]
                            if sequence in shorten_propagation
                            else -1
                        ),
                    }
                    for v in os.listdir(videos_directory)
                    if os.path.isfile(os.path.join(videos_directory, v))
                }
            if not exists_coarse:
                not_used[sequence] = not_used_codes.NO_COARSE.value
                continue
            if len(frames_data[general_view][sequence]) == 0:
                not_used[sequence] = not_used_codes.NO_VIDEOS.value
                continue

            with open(
                assembly_frames_file_path
            ) as assembly_frames_file:  # nusar-2021_action_both_9033-b04d_9033_user_id_2021-02-18_141950
                lines = assembly_frames_file.readlines()
                # map start_frame, which is at annotations_fps, to the corresponding frame in the video, which is at sequence_fps
                start_frame = int(lines[0].split("\t")[0])
                if start_frame % 2 != 0:
                    start_frame -= 1  # Solve problems of desync
                if not propagate_shorten:
                    end_frames = [int(line.split("\t")[1]) for line in lines]
            start_second = start_frame / annotations_fps

            no_assembly = {v: False for v in frames_data[general_view][sequence]}
            for v in frames_data[general_view][sequence]:
                video_path = os.path.join(videos_directory, v)
                video = cv2.VideoCapture(video_path)
                max_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
                video.release()
                max_second = max_frame / sequence_fps

                if start_second > max_second:
                    no_assembly[v] = True
                    continue
                max_annotation_frame = max_second * annotations_fps
                if not propagate_shorten and end_frames[-1] > max_annotation_frame:
                    for end_frame in reversed(end_frames):
                        if end_frame < max_annotation_frame:
                            frames_data[general_view][sequence][v][
                                "new_end_frame"
                            ] = end_frame
                            break

                frames_data[general_view][sequence][v]["first_frame"] = start_frame

            all_no_assembly = all(no_assembly.values())
            any_no_assembly = any(no_assembly.values())
            if all_no_assembly:
                not_used[sequence] = not_used_codes.NO_ASSEMBLY.value
            elif any_no_assembly:
                for view, view_data in frames_data[general_view][sequence].items():
                    if view_data == -1:
                        not_used[sequence + "/" + view] = (
                            not_used_codes.NO_ASSEMBLY_VIDEO.value
                        )

            if not propagate_shorten and not all_no_assembly:
                new_end_frames = [
                    frames["new_end_frame"]
                    for frames in frames_data[general_view][sequence].values()
                    if frames["new_end_frame"] != -1
                ]
                if edit_all_if_too_short[general_view] and len(new_end_frames) > 0:
                    new_end_frame = min(new_end_frames)
                    for v in frames_data[general_view][sequence]:
                        frames_data[general_view][sequence][v][
                            "new_end_frame"
                        ] = new_end_frame
                    shorten_propagation[sequence] = (new_end_frame, set([general_view]))
            elif propagate_shorten:
                shorten_propagation[sequence][1].push(general_view)

    for sequence, shorten_data in shorten_propagation.items():
        for general_view in set(general_views) - shorten_data[1]:
            for v in frames_data[general_view][sequence]:
                frames_data[general_view][sequence][v]["new_end_frame"] = shorten_data[
                    0
                ]

    # Remove remaining not used sequences
    for sequence in not_used:
        for general_view in general_views:
            if sequence in frames_data[general_view]:
                del frames_data[general_view][sequence]

    print("Cropping assembly videos")
    for general_view in general_views:
        print(f"_Processing {general_view} view")
        for sequence in tqdm(list(frames_data[general_view].keys())):
            if sequence in not_used:
                del frames_data[general_view][sequence]
                continue

            videos_directory = os.path.join(sequences_paths[general_view], sequence)
            assembly_videos_directory = os.path.join(
                assembly_directories[general_view], sequence
            )
            if not os.path.exists(assembly_videos_directory):
                os.makedirs(assembly_videos_directory)
            for v in list(frames_data[general_view][sequence].keys()):
                if sequence + "/" + v in not_used:
                    del frames_data[general_view][sequence][v]
                    continue

                assembly_video_path = os.path.join(assembly_videos_directory, v)
                if os.path.exists(assembly_video_path):
                    continue

                # crop the video to the assembly
                video_path = os.path.join(videos_directory, v)
                start_second = (
                    frames_data[general_view][sequence][v]["first_frame"]
                    / annotations_fps
                )
                ffmpeg.input(video_path, ss=start_second).output(
                    assembly_video_path, vcodec="h264_nvenc", loglevel="quiet"
                ).run(
                    # overwrite_output=True
                )

    print("Creating offet files")
    for general_view in general_views:
        with open(offsets_paths[general_view], "w") as offsets:
            offsets.write(",".join(first_lines["offsets"]) + "\n")
            for i, sequence in enumerate(frames_data[general_view]):
                for v in frames_data[general_view][sequence]:
                    offsets.write(
                        f"{i},"
                        + f"{sequence + '/' + v},"
                        + f"{str(frames_data[general_view][sequence][v]['first_frame'])},"
                        + f"{str(frames_data[general_view][sequence][v]['new_end_frame']) if frames_data[general_view][sequence][v]['new_end_frame'] != -1 else '-'}\n"
                    )

    print("_Listing not used videos")
    with open(not_used_files_path, "w") as o:
        for name in not_used:
            o.write(f"{name},{not_used[name]}\n")
else:
    frames_data = {
        "ego": get_view_frames_data(offsets_paths["ego"]),
    }

if remove_disassembly_rows["labels"]:
    print("\nRemoving not used sequences from labels")
    print("_Processing coarse labels")
    count = 0
    disassembly_count = 0
    shorten_count = 0
    directories = os.listdir(
        os.path.join(annotations_directories["coarse"], "coarse_labels")
    )
    for coarse_sequence in directories:
        sequence = "_".join(coarse_sequence.split("_")[1:])[:-4]
        if sequence not in frames_data["ego"]:
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

        coarse_sequence_path = os.path.join(
            annotations_directories["coarse"], "coarse_labels/", coarse_sequence
        )
        assembly_coarse_sequence_path = os.path.join(
            assembly_annotations_directories["coarse"], "coarse_labels"
        )
        new_end_frames = [
            frames_data["ego"][sequence][v]["new_end_frame"]
            for v in frames_data["ego"][sequence]
            if frames_data["ego"][sequence][v]["new_end_frame"] != -1
        ]
        if len(new_end_frames) == 0:
            shutil.copy(
                coarse_sequence_path,
                assembly_coarse_sequence_path,
            )
        else:
            new_end_frame = (
                max(new_end_frames)
                if len(new_end_frames) != len(frames_data["ego"][sequence])
                else sys.maxsize
            )
            with open(coarse_sequence_path) as f:
                lines = f.readlines()
            with open(assembly_coarse_sequence_path, "w") as f:
                for line in lines:
                    end_frame = line.split("\t")[1]
                    if end_frame <= new_end_frame:
                        f.write(line)
            shorten_count += 1
    print(
        f"- Removed sequences ({len(directories) - disassembly_count} -> {len(directories) - disassembly_count - count}): {count}"
    )
    print(f"- Shorten sequences: {shorten_count}")

    print("_Processing skills labels")
    for skill_level_file in os.listdir(annotations_directories["skills"]):
        count = 0
        with open(
            os.path.join(annotations_directories["skills"], skill_level_file)
        ) as f:
            lines = f.readlines()

        with open(
            os.path.join(assembly_annotations_directories["skills"], skill_level_file),
            "w",
        ) as f:
            for line in lines:
                line = line.strip()
                if line in frames_data["ego"]:
                    f.write(line + "\n")
                else:
                    count += 1
            print(
                f"- Removed sequences of level {skill_level_file[:-4].split('_')[1]} ({len(lines)} -> {len(lines) - count}): {count}"
            )

if remove_disassembly_rows["splits_actions"]:
    print("\nRemoving disassembly rows from splits")
    assembly_actions = {annotation_type: set() for annotation_type in action_getter}
    split_lines = {}
    for split in splits:
        print(f"Processing {split}")
        with open(os.path.join(annotations_directories["coarse"], "actions.csv")) as f:
            lines = f.readlines()
            coarse_actions = {
                action_getter["coarse"]["cls"](l): action_getter["coarse"]["id"](l)
                for l in lines[1:]
            }
        with open(os.path.join(annotations_directories["fine"], split + ".csv")) as f:
            lines = f.readlines()

        split_lines[split] = [",".join(first_lines["splits"]["fine"][split]) + "\n"]
        count = -1
        last_id = -1
        for line in lines[1:]:
            line = line.strip().split(",")
            id = int(line[0])
            start_frame = int(line[2])
            end_frame = int(line[3])

            if "_challenge" not in split:
                name, view = line[1].split("/")
            else:
                name = line[1]
                view = (
                    (
                        list(frames_data["ego"][name].keys())[0]
                        if name in frames_data["ego"]
                        else None
                    )
                    if name in frames_data["ego"] and len(frames_data["ego"][name]) > 0
                    else None
                )

            if (
                name not in frames_data["ego"]
                or view not in frames_data["ego"][name]
                or start_frame < frames_data["ego"][name][view]["first_frame"]
                or frames_data["ego"][name][view]["new_end_frame"] != -1
                and end_frame > frames_data["ego"][name][view]["new_end_frame"]
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
                ) as o:
                    assembly_actions["coarse"].update(
                        [coarse_actions[line.split("\t")[2]] for line in o.readlines()]
                    )
            split_lines[split].append(f"{count},{','.join(line[1:])}\n")

    print("\nProcessing actions")
    assembly_actions_mapping = {}
    for annotation_type in action_getter:
        with open(
            os.path.join(annotations_directories[annotation_type], "actions.csv")
        ) as f:
            lines = f.readlines()
            actions = set([action_getter[annotation_type]["id"](l) for l in lines[1:]])

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
            actions_to_remove = actions - assembly_actions[annotation_type]
            ordered_actions_to_remove = list(actions_to_remove)
            ordered_actions_to_remove.sort()
            print(
                f"{annotation_type.title()} actions to remove ({len(actions)} -> {len(assembly_actions[annotation_type])}): {len(ordered_actions_to_remove)}"
            )
            print(ordered_actions_to_remove)
            assembly_actions_mapping[annotation_type] = {
                id: id for id in assembly_actions[annotation_type]
            }
            if remove_disassembly_rows["map_actions"]:
                print("Mapping actions")

                ordered_assembly_actions = list(assembly_actions[annotation_type])
                ordered_assembly_actions.sort()

                with open(actions_mapping_path[annotation_type], "w") as amf:
                    amf.write(",".join(first_lines["actions_mapping"]) + "\n")
                    for i, id in enumerate(ordered_assembly_actions):
                        assembly_actions_mapping[annotation_type][id] = i
                        amf.write(f"{i},{id},{i}\n")

            for line in lines[1:]:
                divided_line = line.strip().split(",")
                if (
                    action_getter[annotation_type]["id"](divided_line, True)
                    not in actions_to_remove
                ):
                    f.write(
                        str(count)
                        + ","
                        + str(
                            assembly_actions_mapping[annotation_type][
                                int(divided_line[1])
                            ]
                            if remove_disassembly_rows["map_actions"]
                            else divided_line[1]
                        )
                        + ","
                        + ",".join(divided_line[2:])
                        + "\n"
                    )
                    count += 1
                else:
                    o.write(f"{','.join(divided_line[1:])}\n")

        for split in splits:
            with open(
                os.path.join(assembly_annotations_directories["fine"], split + ".csv"),
                "w",
            ) as f:
                if remove_disassembly_rows["map_actions"]:
                    print(f"Mapping {split} actions")
                    f.write(split_lines[split][0])
                    for line in split_lines[split][1:]:
                        divided_line = line.split(",")
                        a = int(divided_line[4])
                        b = assembly_actions_mapping["fine"][a]
                        if a == 542:
                            pass
                        f.write(
                            ",".join(divided_line[:4] + [str(b)] + divided_line[5:])
                        )
                else:
                    f.writelines(split_lines[split])
