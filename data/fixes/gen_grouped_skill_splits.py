import os
import csv
import sys

from utils.constants import first_lines


def write_split_challenge(source_path, destination_path, split):
    with open(os.path.join(source_path, f"{split}.csv"), "r") as f:
        csv_reader = csv.reader(f)
        with open(
            os.path.join(destination_path, f"{split}_challenge.csv"), "w", newline=""
        ) as f:
            csv_writer = csv.writer(f)
            # Write header of csv_reader
            csv_writer.writerow(next(csv_reader))
            id = -1
            for row in csv_reader:
                if row[0] != id:
                    id = row[0]
                    sequence = row[1].split("/")[0]
                    csv_writer.writerow([id, sequence] + row[2:])


def create_grouped_csv(
    splits,
    coarse_annotations_path,
    skill_evaluation_path,
    sequences_path,
    annotations_fps,
    max_group_duration,
    min_group_duration,
    tolerance_group_duration,
    join_duration,
    skill_aggregations,
    additional_splits,
):
    coarse_labels_dir = os.path.join(coarse_annotations_path, "coarse_labels")
    coarse_splits_dir = os.path.join(coarse_annotations_path, "coarse_splits")
    skill_files = [
        os.path.join(skill_evaluation_path, f"skill_labels/skill_{i}.txt")
        for i in range(1, 6)
    ]

    # Mapping file to skill level based on the skill_labels_path files
    file_to_skill_level = {}
    for skill_level, skill_file in enumerate(skill_files, start=1):
        with open(skill_file, "r") as f:
            files = f.read().splitlines()
            for view in files:
                file_to_skill_level[view] = skill_aggregations[skill_level]

    # Process each split file
    with open(
        os.path.join(coarse_annotations_path, "joint_labels.csv"), "w", newline=""
    ) as joint_labels_file:
        joint_labels_writer = csv.writer(joint_labels_file)
        joint_labels_writer.writerow(first_lines["joint_coarse_actions"])
        joint_id = 0
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
                coarse_id = 0
                for sequence in file_list:
                    coarse_file_path = os.path.join(coarse_labels_dir, sequence)
                    sequence = "_".join(sequence.split("_")[1:])[:-4]
                    if not os.path.isfile(coarse_file_path):
                        continue

                    with open(coarse_file_path, "r") as coarse_f:
                        actions = [
                            line.strip().split("\t") for line in coarse_f.readlines()
                        ]
                        actions = [
                            (int(start), int(end), action)
                            for start, end, action in actions
                        ]

                    # Join actions that are too short and write the joint actions to CSV
                    joint_actions = []
                    if len(actions) > 1:
                        i = 0
                        while i < len(actions):
                            action = actions[i]
                            current_duration = (action[1] - action[0]) / annotations_fps

                            if current_duration < join_duration:
                                # Join the action to the shortest action among the previous and the next one
                                # Check if the previous action was joined
                                joint = False
                                if i == 0:
                                    # If the current action is the first one, there isn't a previous action
                                    previous_action_duration = sys.maxsize
                                elif (
                                    len(joint_actions) > 0
                                    and joint_actions[-1][1] == action[0]
                                ):
                                    previous_action_duration = (
                                        joint_actions[-1][1] - joint_actions[-1][0]
                                    )
                                    joint = True
                                else:
                                    previous_action_duration = (
                                        actions[i - 1][1] - actions[i - 1][0]
                                    )

                                if i == len(actions) - 1:
                                    # If the current action is the last one, there isn't a next action
                                    next_action_duration = sys.maxsize
                                else:
                                    next_action_duration = (
                                        actions[i + 1][1] - actions[i + 1][0]
                                    )

                                # Check if the current action should be joined with the previous or next one
                                if previous_action_duration < next_action_duration:
                                    if joint:
                                        # If the previous action was joined, update the end frame
                                        joint_actions[-1][1] = action[1]
                                    else:
                                        joint_actions.append(
                                            [
                                                actions[i - 1][0],
                                                action[1],
                                            ]
                                        )
                                else:
                                    # Find the sequence of actions to join
                                    j = i + 1
                                    while (
                                        j < len(actions)
                                        and (actions[j][1] - action[0])
                                        / annotations_fps
                                        < join_duration
                                    ):
                                        j += 1

                                    if j < len(actions):
                                        joint_actions.append([action[0], actions[j][1]])
                                    elif (
                                        len(joint_actions) > 0
                                        and joint_actions[-1][1] == action[0]
                                    ):
                                        # If there are no more actions to join, update the end frame of the last action joined
                                        joint_actions[-1][1] = actions[j - 1][1]
                                    elif i - 1 >= 0:
                                        # If there are no more actions to join and the previous action was not joined, join the previous and the actual sequence of actions
                                        joint_actions.append(
                                            [actions[i - 1][0], actions[j - 1][1]]
                                        )
                                    else:
                                        # If there are no more actions to join and no actions were joined, join the actual sequence of actions
                                        joint_actions.append(
                                            [action[0], actions[j - 1][1]]
                                        )
                                    i = j  # Skip the actions that were joined
                            i += 1

                        for joint_action in joint_actions:
                            joint_labels_writer.writerow(
                                [
                                    joint_id,
                                    sequence,
                                    joint_action[0],
                                    joint_action[1],
                                ]
                            )
                            joint_id += 1

                        if len(joint_actions) > 0:
                            i = 0
                            j = 0
                            # Remove actions that are contained in the joint actions and replace them with correspondent joint action
                            last_end_frame = -1
                            while i < len(actions) and j < len(joint_actions):
                                if (
                                    actions[i][0] >= joint_actions[j][0]
                                    and actions[i][0] < joint_actions[j][1]
                                ):
                                    last_end_frame = actions.pop(i)[1]
                                else:
                                    i += 1

                                if last_end_frame == joint_actions[j][1]:
                                    actions.insert(
                                        i,
                                        (joint_actions[j][0], joint_actions[j][1], ""),
                                    )
                                    i += 1
                                    j += 1
                                    last_end_frame = -1

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
                            elif len(groups) > 0:
                                # Try to move the first actions of the current group to the previous until we can add the current action to the current group, considering tollerance for both groups
                                tmp_group = []
                                for tmp_action in current_group:
                                    tmp_group.append(tmp_action)
                                    tmp_duration = (
                                        tmp_group[-1][1] - groups[-1][0][0]
                                    ) / annotations_fps
                                    current_duration = (
                                        end_frame
                                        - (
                                            current_group[len(tmp_group)][0]
                                            if len(tmp_group) < len(current_group)
                                            else start_frame
                                        )
                                    ) / annotations_fps
                                    if (
                                        tmp_duration
                                        <= max_group_duration + tolerance_group_duration
                                        and tmp_duration
                                        >= min_group_duration - tolerance_group_duration
                                        and current_duration
                                        <= max_group_duration + tolerance_group_duration
                                        and current_duration
                                        >= min_group_duration - tolerance_group_duration
                                    ):
                                        groups[-1].extend(tmp_group)
                                        current_group = current_group[
                                            len(tmp_group) :
                                        ] + [action]
                                        break
                                groups.append(current_group)
                                current_group = (
                                    []
                                    if current_group[-1][1] == end_frame
                                    else [action]
                                )
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
                    tmp_group = []
                    for group in groups:
                        group_duration = (group[-1][1] - group[0][0]) / annotations_fps
                        if group_duration >= min_group_duration:
                            if tmp_group:
                                redistributed_groups.append(tmp_group)
                                tmp_group = []
                            redistributed_groups.append(group)
                        else:
                            if not tmp_group:
                                tmp_group = group
                            else:
                                # Attempt to split the group and add part of it to tmp_group
                                for action in group:
                                    new_start_frame = tmp_group[0][0]
                                    new_end_frame = action[1]
                                    combined_duration = (
                                        new_end_frame - new_start_frame
                                    ) / annotations_fps

                                    if (
                                        combined_duration
                                        >= min_group_duration - tolerance_group_duration
                                    ):
                                        redistributed_groups.append(tmp_group)
                                        tmp_group = [action]
                                    else:
                                        tmp_group.append(action)

                    # Append any remaining actions to ensure all groups meet the minimum duration
                    if tmp_group:
                        if len(redistributed_groups) > 0 and (
                            redistributed_groups[-1][-1][1] <= tmp_group[0][0]
                        ):
                            # Merge temp_group with the previous group if it doesn't overlap and keeps order
                            redistributed_groups[-1].extend(tmp_group)
                        else:
                            # Consider merging the last group with the previous one if within tolerance
                            if len(redistributed_groups) > 0:
                                last_group = redistributed_groups[-1]
                                combined_duration = (
                                    tmp_group[-1][1] - last_group[0][0]
                                ) / annotations_fps
                                if (
                                    combined_duration
                                    >= min_group_duration - tolerance_group_duration
                                ):
                                    redistributed_groups.append(tmp_group)
                                else:
                                    redistributed_groups[-1].extend(tmp_group)
                            else:
                                redistributed_groups.append(tmp_group)

                    # Write the redistributed groups to CSV
                    for group in redistributed_groups:
                        start_frame = group[0][0]
                        end_frame = group[-1][1]
                        skill_level = file_to_skill_level[sequence]
                        sequence_dir = os.path.join(sequences_path, sequence)
                        for view in os.listdir(sequence_dir):
                            if view.endswith(".mp4"):
                                csv_writer.writerow(
                                    [
                                        coarse_id,
                                        sequence + "/" + view[:-4],
                                        start_frame,
                                        end_frame,
                                        skill_level,
                                    ]
                                )
                        coarse_id += 1

    # Creates additional splits
    if additional_splits["trainval"]:
        with open(
            os.path.join(skill_evaluation_path, f"skill_splits/train.csv"), "r"
        ) as f:
            train_lines = f.readlines()
        with open(
            os.path.join(skill_evaluation_path, f"skill_splits/validation.csv"), "r"
        ) as f:
            validation_lines = f.readlines()[1:]
        with open(
            os.path.join(skill_evaluation_path, f"skill_splits/trainval.csv"), "w"
        ) as f:
            f.writelines(train_lines)
            f.writelines(validation_lines)
    if additional_splits["validation_challenge"]:
        skill_splits_path = os.path.join(skill_evaluation_path, "skill_splits")
        write_split_challenge(skill_splits_path, skill_splits_path, "validation")
    if additional_splits["test_challenge"]:
        skill_splits_path = os.path.join(skill_evaluation_path, "skill_splits")
        write_split_challenge(skill_splits_path, skill_splits_path, "test")


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
    coarse_annotations_path = "D:/data/annotations/coarse-annotations/"
    skill_evaluation_path = "D:/data/annotations/skill_evaluation/"
    sequences_path = "D:/data/videos/ego_recordings/"
    annotations_fps = 30  # Example FPS
    max_group_duration = 40  # Example max group duration in seconds
    min_group_duration = 30  # Example min group duration in seconds
    tolerance_group_duration = 10  # Tolerance to add to max_group_duration and min_group_duration when considering merging groups
    join_duration = (
        10  # Duration in seconds to consider merging a coarse action to another one
    )
    skill_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
    additional_splits = {
        "trainval": True,
        "validation_challenge": True,
        "test_challenge": True,
    }

    create_grouped_csv(
        splits,
        coarse_annotations_path,
        skill_evaluation_path,
        sequences_path,
        annotations_fps,
        max_group_duration,
        min_group_duration,
        tolerance_group_duration,
        join_duration,
        skill_mapping,
        additional_splits,
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
