import os
import csv

from utils.constants import first_lines


def create_grouped_csv(
    splits,
    coarse_annotations_path,
    skill_evaluation_path,
    sequences_path,
    annotations_fps,
    max_group_duration,
    min_group_duration,
    skill_aggregations,
):
    coarse_labels_dir = os.path.join(coarse_annotations_path, "coarse_labels")
    coarse_splits_dir = os.path.join(coarse_annotations_path, "coarse_splits")
    skill_files = [
        os.path.join(skill_evaluation_path, f"skill_labels/skill_{i}.txt")
        for i in range(1, 6)
    ]
    tolerance_group_duration = 2  # Tolerance to add to max_group_duration when considering merging the last group

    # Mapping file to skill level based on the skill_labels_path files
    file_to_skill_level = {}
    for skill_level, skill_file in enumerate(skill_files, start=1):
        with open(skill_file, "r") as f:
            files = f.read().splitlines()
            for view in files:
                file_to_skill_level[view] = skill_aggregations[skill_level]

    # Process each split file
    for split_file in os.listdir(coarse_splits_dir):
        split = split_file.split("_")[0]
        if split not in splits:
            continue

        split_file_path = os.path.join(coarse_splits_dir, split_file)
        if not os.path.isfile(split_file_path):
            continue

        with open(split_file_path, "r") as split_f:
            file_list = [line.split("\t")[0] for line in split_f.readlines()]

        output_csv_path = os.path.join(
            skill_evaluation_path, f"skill_splits/{split}.csv"
        )
        with open(output_csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(first_lines["splits"]["skill"][split])
            id = 0
            for sequence in file_list:
                coarse_file_path = os.path.join(coarse_labels_dir, sequence)
                if not os.path.isfile(coarse_file_path):
                    continue

                with open(coarse_file_path, "r") as coarse_f:
                    actions = [
                        line.strip().split("\t") for line in coarse_f.readlines()
                    ]
                    actions = [
                        (int(start), int(end), action) for start, end, action in actions
                    ]

                # Group actions based on max and min group duration
                groups = []
                current_group = []

                for action in actions:
                    start_frame, end_frame, _ = action

                    if current_group:
                        total_duration = (
                            end_frame - current_group[0][0]
                        ) / annotations_fps
                        if total_duration <= max_group_duration:
                            current_group.append(action)
                        else:
                            groups.append(current_group)
                            current_group = [action]
                    else:
                        current_group = [action]

                # Append the last group
                if current_group:
                    groups.append(current_group)

                # Redistribute actions to ensure all groups meet min_group_duration without overlapping
                redistributed_groups = []
                temp_group = []

                for group in groups:
                    group_duration = (group[-1][1] - group[0][0]) / annotations_fps
                    if group_duration >= min_group_duration:
                        if temp_group:
                            redistributed_groups.append(temp_group)
                            temp_group = []
                        redistributed_groups.append(group)
                    else:
                        if not temp_group:
                            temp_group = group
                        else:
                            # Attempt to split the group and add part of it to temp_group
                            for action in group:
                                new_start_frame = temp_group[0][0]
                                new_end_frame = action[1]
                                combined_duration = (
                                    new_end_frame - new_start_frame
                                ) / annotations_fps

                                if combined_duration <= max_group_duration:
                                    temp_group.append(action)
                                else:
                                    redistributed_groups.append(temp_group)
                                    temp_group = [action]

                # Append any remaining actions to ensure all groups meet the minimum duration
                if temp_group:
                    if len(redistributed_groups) > 0 and (
                        redistributed_groups[-1][-1][1] <= temp_group[0][0]
                    ):
                        # Merge temp_group with the previous group if it doesn't overlap and keeps order
                        redistributed_groups[-1].extend(temp_group)
                    else:
                        # Consider merging the last group with the previous one if within tolerance
                        if len(redistributed_groups) > 0:
                            last_group = redistributed_groups[-1]
                            combined_duration = (
                                temp_group[-1][1] - last_group[0][0]
                            ) / annotations_fps
                            if (
                                combined_duration
                                <= max_group_duration + tolerance_group_duration
                            ):
                                redistributed_groups[-1].extend(temp_group)
                            else:
                                redistributed_groups.append(temp_group)
                        else:
                            redistributed_groups.append(temp_group)

                # Write the redistributed groups to CSV
                sequence = "_".join(sequence.split("_")[1:])[:-4]
                for group in redistributed_groups:
                    start_frame = group[0][0]
                    end_frame = group[-1][1]
                    skill_level = file_to_skill_level[sequence]
                    sequence_dir = os.path.join(sequences_path, sequence)
                    for view in os.listdir(sequence_dir):
                        if view.endswith(".mp4"):
                            csv_writer.writerow(
                                [
                                    id,
                                    sequence + "/" + view[:-4],
                                    start_frame,
                                    end_frame,
                                    skill_level,
                                ]
                            )
                    id += 1


# Find the row with the same start_frame of actual_line[2]. Then, check if all the rows until the row with the same end_frame of actual_line[3] have are contained in the actual_line[2] and actual_line[3] range
def update_skill_statistics(
    sequence,
    actual_range,
    coarse_sequence,
    actual_start_frame,
    actual_end_frame,
    statistics,
):
    statistics["mean"]["duration"] += actual_range
    if (
        statistics["min"]["duration"][0] == 0
        or actual_range < statistics["min"]["duration"][0]
    ):
        statistics["min"]["duration"] = (actual_range, sequence, actual_start_frame)
    if actual_range > statistics["max"]["duration"][0]:
        statistics["max"]["duration"] = (actual_range, sequence, actual_start_frame)

    start_frame_found = False
    actions_per_group = 0
    for row in coarse_sequence:
        tmp_start_frame = int(row[0])
        tmp_end_frame = int(row[1])
        if tmp_start_frame == actual_start_frame:
            start_frame_found = True
        if start_frame_found:
            actions_per_group += 1
            if tmp_start_frame < actual_start_frame or tmp_end_frame > actual_end_frame:
                print(
                    f"Error in {split} split: rows are not contained in the actual_line[2] and actual_line[3] range"
                )
                return False
        if tmp_end_frame == actual_end_frame:
            break

    statistics["mean"]["actions"] += actions_per_group
    if (
        statistics["min"]["actions"][0] == 0
        or actions_per_group < statistics["min"]["actions"][0]
    ):
        statistics["min"]["actions"] = (actions_per_group, sequence, actual_start_frame)
    if actions_per_group > statistics["max"]["actions"][0]:
        statistics["max"]["actions"] = (actions_per_group, sequence, actual_start_frame)

    return True


# Check if, for every line with different id, the start_frame is greater than the end_frame of the previous line
def check_split(split, skill_evaluation_path, coarse_labels_path):
    with open(
        os.path.join(skill_evaluation_path, f"skill_splits/{split}.csv"), "r"
    ) as skill_split_file:
        skill_split = list(csv.reader(skill_split_file))

    last_id = -1
    statistics = {
        "mean": {
            "duration": 0,
            "actions": 0,
        },
        "min": {
            "duration": (0, "", 0),
            "actions": (0, "", 0),
        },
        "max": {
            "duration": (0, "", 0),
            "actions": (0, "", 0),
        },
    }
    skill_groups = 0
    for i in range(1, len(skill_split)):
        actual_line = skill_split[i]
        actual_sequence = actual_line[1].split("/")[0]
        actual_start_frame = int(actual_line[2])
        actual_end_frame = int(actual_line[3])
        actual_range = actual_end_frame - actual_start_frame

        if last_id == -1:
            coarse_sequence_path = os.path.join(
                coarse_labels_path, "assembly_" + actual_sequence + ".txt"
            )
            with open(coarse_sequence_path, "r") as coarse_sequence_file:
                coarse_sequence = list(csv.reader(coarse_sequence_file, delimiter="\t"))
            update_skill_statistics(
                actual_sequence,
                actual_range,
                coarse_sequence,
                actual_start_frame,
                actual_end_frame,
                statistics,
            )
            skill_groups += 1
            last_id = int(actual_line[0])
        if int(skill_split[i][0]) != last_id:
            last_id = int(actual_line[0])
            previous_line = skill_split[i - 1]
            if actual_sequence == previous_line[1].split("/")[0]:
                if actual_start_frame < int(previous_line[3]):
                    print(
                        f"Error in {split} split: start_frame is greater than the end_frame of the previous line"
                    )
                    return None

                update_skill_statistics(
                    actual_sequence,
                    actual_range,
                    coarse_sequence,
                    actual_start_frame,
                    actual_end_frame,
                    statistics,
                )
                skill_groups += 1
            else:
                coarse_sequence_path = os.path.join(
                    coarse_labels_path, "assembly_" + actual_sequence + ".txt"
                )
                with open(coarse_sequence_path, "r") as coarse_sequence_file:
                    coarse_sequence = list(
                        csv.reader(coarse_sequence_file, delimiter="\t")
                    )

    for key in statistics["mean"].keys():
        statistics["mean"][key] /= skill_groups
    for stat_key in statistics.keys():
        if type(statistics[stat_key]["duration"]) == tuple:
            tmp_stat = list(statistics[stat_key]["duration"])
            tmp_stat[0] /= annotations_fps
            statistics[stat_key]["duration"] = tmp_stat
        else:
            statistics[stat_key]["duration"] /= annotations_fps

    return statistics


if __name__ == "__main__":
    splits = ["train", "validation", "test"]  # ["train", "validation", "test"]
    coarse_annotations_path = "D:/data/assembly/annotations/coarse-annotations/"
    skill_evaluation_path = "D:/data/assembly/annotations/skill_evaluation/"
    sequences_path = "D:/data/assembly/videos/ego_recordings/"
    annotations_fps = 30  # Example FPS
    max_group_duration = 90  # Example max group duration in seconds
    min_group_duration = 60  # Example min group duration in seconds
    skill_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}

    create_grouped_csv(
        splits,
        coarse_annotations_path,
        skill_evaluation_path,
        sequences_path,
        annotations_fps,
        max_group_duration,
        min_group_duration,
        skill_mapping,
    )

    for split in splits:
        statistics = check_split(
            split,
            skill_evaluation_path,
            os.path.join(coarse_annotations_path, "coarse_labels"),
        )
        print("\nSplit:", split)
        for stat, value in statistics.items():
            print(f"{stat}:")
            for key, stat_value in value.items():
                print(f"- {key.title()}: {stat_value}")