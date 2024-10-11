import os
from itertools import cycle
import numpy as np
from statistics import mean

from utils.constants import first_lines
from utils.classes import Sequence


def map_skill_level(skill_aggregations, skill_level):
    for skill_level_aggregation in skill_aggregations:
        if skill_level in skill_aggregations[skill_level_aggregation]:
            return skill_level_aggregation
    return None


def get_splits_distributions(
    splits_divisions: dict[str, str], data: dict[str, dict[Sequence, dict[str, any]]]
):
    global people_per_split

    skill_levels = {
        split: {skill_level: 0 for skill_level in skill_aggregations}
        for split in people_per_split
    }
    for person in splits_divisions:
        for sequence in data[person]:
            skill_level = data[person][sequence]["skill_level"]
            skill_levels[splits_divisions[person]][skill_level] += 1

            """splits_distributions[splits_divisions[person]]["duration"] += data[person][sequence][
                "end_frame"
            ]
        splits_distributions[splits_divisions[person]]["duration"] //= sequence_fps"""

    # Check if the distribution is balanced checking that the difference between the maximum and minimum is less than the balance_tollerances
    splits_distributions = {
        split: {
            "skills": tuple(skill_levels[split].values()),
            # "duration": 0,
        }
        for split in people_per_split
    }

    print("_Videos per split/skill level:")
    for split in people_per_split:
        print(f"- {split}: {splits_distributions[split]['skills']}")
    """print("_Duration:")
    for split in people_per_split:
        print(f"- {split}: {splits_distributions[split]['duration']}")"""

    return splits_distributions


def check_balance(splits_distributions):
    global people_per_split

    total_videos = {}
    skill_dristributions_diffs = []
    # durations = []
    for split in splits_distributions:
        # durations.append(splits_distributions[split]["duration"])
        total_videos[split] = sum(splits_distributions[split]["skills"])

    to_print = {split: [] for split in people_per_split}
    for i in range(len(skill_aggregations)):
        skills_rel = []
        for split in people_per_split:
            skills_rel.append(
                splits_distributions[split]["skills"][i] / total_videos[split]
            )
            to_print[split].append(round(skills_rel[-1] * 100, 2))
        skill_dristributions_diffs.append(max(skills_rel) - min(skills_rel))
        """skill_dristributions_diffs.append(
            max(splits_distributions[split]["skills"])
            - min(splits_distributions[split]["skills"])
            < balance_tollerances["skills"]
        )"""

    print("_Percentages:")
    for split in to_print:
        print(f"- {split}: {tuple(to_print[split])}")
    return {
        "processed_info": {
            "skills_rel": skill_dristributions_diffs,
            # "duration": durations,
        },
        "balance": {
            # "skills": all(skill_dristributions_diffs),
            "skills_rel": all(
                [
                    diff < balance_tollerances["skills_rel"]
                    for diff in skill_dristributions_diffs
                ]
            ),
            # "duration": max(durations) - min(durations) < balance_tollerances["duration"],
        },
    }


def get_nearest_person(
    splits_divisions: dict[str, str],
    data: dict[str, dict[Sequence, dict[str, any]]],
    to_fix: tuple[str, int],
    target: tuple[str, int],
):
    global people_per_split

    to_fix_person = ("", -10)
    target_person = ("", -10)
    for person in splits_divisions:
        if splits_divisions[person] == to_fix[0]:
            tmp_mean = mean(
                [data[person][sequence]["skill_level"] for sequence in data[person]]
            )
            if abs(tmp_mean - to_fix[1]) < abs(to_fix_person[1] - to_fix[1]):
                to_fix_person = (person, tmp_mean)
        elif splits_divisions[person] == target[0]:
            tmp_mean = mean(
                [data[person][sequence]["skill_level"] for sequence in data[person]]
            )
            if abs(tmp_mean - target[1]) < abs(target_person[1] - target[1]):
                target_person = (person, tmp_mean)

    return to_fix_person[0], target_person[0]


def move_person(
    splits_order: dict[str, list[Sequence]],
    splits_divisions,
    data: dict[str, dict[Sequence, dict[str, any]]],
    person: str,
    split: str,
):
    sequences = list(data[person].keys())
    splits_order[splits_divisions[person]] = list(
        filter(lambda x: x not in sequences, splits_order[splits_divisions[person]])
    )
    splits_order[split] += sequences
    splits_divisions[person] = split
    return True


def write_split_challenge(split: str):
    global target_splits_directory

    with open(
        os.path.join(target_splits_directory, f"{split}_challenge.csv"), "w"
    ) as f:
        f.write(",".join(first_lines["splits"][split]) + "\n")
        id = 0
        for sequence in splits_order[split]:
            person = sequence.person
            for i, row_values in enumerate(data[person][sequence]["values"]):
                f.write(f"{i},{sequence},{','.join(row_values)}\n")
                id += 1


annotations_directories = {
    "fine": "D:/data/assembly/annotations/action_anticipation/cropped",  # Splits directory
    "skills": "D:/data/assembly/annotations/skill_labels/",  # Skills directory
}
target_splits_directory = "D:/data/assembly/annotations/action_anticipation/"
sequence_fps = 15

people_per_split = {
    "train": 30,
    "validation": 9,
    "test": 9,
}
additional_splits = {
    "trainval": True,
    "validation_challenge": True,
    "test_challenge": True,
}
skill_aggregations = {
    1: set([1, 2]),
    2: set([3, 4]),
    3: set([5]),
}
balance_tollerances = {
    # "skills": 20,  # Number of elements that a skill level in a split can differ from the others
    "skills_rel": 0.1,  #
    # "duration": 20,  # Seconds
}

# Create a dictionary of the levels for all people
data: dict[str, dict[Sequence, dict[str, any]]] = {}
for skill_level_file in os.listdir(annotations_directories["skills"]):
    with open(
        os.path.join(annotations_directories["skills"], skill_level_file), "r"
    ) as f:
        for line in f:
            sequence = Sequence(line.strip())
            if (
                sequence
                == "nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253"
            ):
                pass
            person = sequence.person
            skill_level = int(skill_level_file[:-4].split("_")[1])
            if person not in data:
                data[person] = {}
            data[person][sequence] = {
                "skill_level": map_skill_level(skill_aggregations, skill_level),
            }

actual_splits_order: dict[str, list[str]] = {}
actual_splits_divisions: dict[str, dict[str, list[str]]] = {}
for split in people_per_split:
    actual_splits_order[split] = []
    actual_splits_divisions = {}
    with open(os.path.join(annotations_directories["fine"], f"{split}.csv"), "r") as f:
        lines = f.readlines()
    last_id = -1
    last_sequence = ["", -1]  # (sequence, end_frame)
    for line in lines[1:]:
        id = int(line.split(",")[0])
        if id != last_id:
            line = line.strip()
            divided_line = line.split(",")
            sequence, view = divided_line[1].split("/")
            sequence = Sequence(sequence)
            person = sequence.person
            end_frame = int(divided_line[3])
            values = divided_line[2:]
            last_id = id

            data[person][sequence]["views"] = [view]

            if person not in actual_splits_divisions:
                actual_splits_divisions[person] = {s: [] for s in people_per_split}
                actual_splits_divisions[person][split] = [sequence]
            elif actual_splits_divisions[person][split][-1] != sequence:
                actual_splits_divisions[person][split].append(sequence)

            if "values" not in data[person][sequence]:
                data[person][sequence]["values"] = []
            data[person][sequence]["values"].append(values)

            if sequence != last_sequence[0]:
                actual_splits_order[split].append(sequence)
                if last_sequence != ["", -1]:
                    data[last_sequence[0].person][last_sequence[0]]["end_frame"] = (
                        last_sequence[1]
                    )
                last_sequence[0] = sequence
            last_sequence[1] = end_frame
        else:
            divided_line = line.strip().split(",")
            view = divided_line[1].split("/")[1]
            data[person][sequence]["views"].append(view)

    if last_sequence[0] != "":
        data[last_sequence[0].person][last_sequence[0]]["end_frame"] = last_sequence[1]

"""
data = {
    p1: { s1: { skill_level: 2, end_frame: 10, values: [[1,2,4],[5,6,3],[6,10,5]], },
        s2: { skill_level: 3, end_frame: 6, values: [[1,4,4],[5,6,4]], },
    },
    p2: { s3: { skill_level: 4, end_frame: 21, values: [[1,7,8],[5,6,3],[11,21,9]], },
        s4: { skill_level: 4, end_frame: 15, values: [[4,6,3],[5,7,4],[6,15,5]], },
        s5: { skill_level: 5, end_frame: 8, values: [[1,6,8],[6,7,1],[7,8,2]], }, },
    p3: { s6: { skill_level: 4, end_frame: 5, values: [[0,2,3],[2,4,4],[3,5,2]], }, }
}
actual_splits_order = {
    train: [s1, s2, s3],
    validation: [s4, s5,],
    test: [s6,],
}
actual_splits_divisions = {
    p1: { train: [s1, s2], validation: [], test: [], },
    p2: { train: [s3], validation: [s4, s5], test: [], },
    p3: { train: [], validation: [], test: [s6], },
}
splits_order = {
    train: [s3, s4, s5,],
    validation: [s1, s2,],
    test: [s6,],
}
splits_divisions = {
    p1: "validation",
    p2: "train",
    p3: "test",
}

# steps:
1) Sort in descending order of people by total duration of their videos
2) Place people in order to follow the people_per_split
Loop until is balanced:
    3) Compute the distribution of every skill level in the splits
    4) Check if the distribution is balanced
    5) If not, move the person with the highest skill level to the split with the lowest skill level, exchanging with a person with the skill level with many people
"""

# Sort in descending order of people by total duration of their videos
people = list(data.keys())
people.sort(key=lambda p: sum([data[p][s]["end_frame"] for s in data[p]]), reverse=True)

# Place people in order to follow the people_per_split
splits_order: dict[str, list[Sequence]] = {split: [] for split in people_per_split}
splits_counters: dict[str, int] = {split: 0 for split in people_per_split}
splits_divisions: dict[str, str] = {person: None for person in people}
splits = cycle(people_per_split.keys())
for person in people:
    count = 0
    while count < len(people_per_split):
        split = next(splits)
        count += 1
        if splits_counters[split] < people_per_split[split]:
            splits_order[split] += data[person].keys()
            splits_divisions[person] = split
            splits_counters[split] += 1
            break

again = True
save = True
splits_order_archive: list[dict[str, list[Sequence]]] = []
while again:
    print("Distribution n.", len(splits_order_archive))
    splits_order_archive.append(splits_order)

    splits_distribution = get_splits_distributions(splits_divisions, data)
    balance_analysis = check_balance(splits_distribution)
    print()
    if all(balance_analysis["balance"].values()):
        print("Splits are balanced.")
        again = False
        continue

    print("Splits are not balanced. Do you want to:")
    print("- Confirm this distribution anyway (1)")
    print("- Exit without saving (2)")
    print("- Come back to a previous one (3)")
    print("- Try another distribution (other)")
    answer = input()

    if answer == "1":
        print("Distribution confirmed")
        again = False
        continue
    elif answer == "2":
        print("Execution terminated")
        again = False
        save = False
        continue
    elif answer == "3":
        right = False
        while not right:
            print("Insert the number of the distribution you want to come back:")
            answer = input()
            if answer.isnumeric():
                answer = int(answer)
                if answer < len(splits_order_archive):
                    splits_order = splits_order_archive[answer]
                    splits_order_archive = splits_order_archive[:answer]
                    right = True
                else:
                    print("Invalid number.")
            else:
                print("Invalid input.")
        continue

    skill_level_to_fix = np.argmax(
        balance_analysis["processed_info"]["skills_rel"]
    )  # 0 based
    skill_level_target = np.argmin(
        balance_analysis["processed_info"]["skills_rel"]
    )  # 0 based

    skills_split = {
        split: splits_distribution[split]["skills"][skill_level_to_fix]
        for split in splits_distribution
    }
    split_to_fix = max(skills_split, key=skills_split.get)
    skills_split = {
        split: splits_distribution[split]["skills"][skill_level_target]
        for split in splits_distribution
    }
    split_target = min(skills_split, key=skills_split.get)

    to_fix_person, target_person = get_nearest_person(
        splits_divisions,
        data,
        (split_to_fix, skill_level_to_fix),
        (split_target, skill_level_target),
    )

    move_person(splits_order, splits_divisions, data, to_fix_person, split_target)
    move_person(splits_order, splits_divisions, data, target_person, split_to_fix)

if save:
    # Update is_shared flag in values
    toys = {
        split: set([sequence.toy for sequence in splits_order[split]])
        for split in splits_order
    }
    shared_toys = toys["train"] & toys["validation"] | toys["train"] & toys["test"]
    for person in data:
        for sequence in data[person]:
            toy = sequence.toy
            for i in range(len(data[person][sequence]["values"])):
                data[person][sequence]["values"][i][-2] = (
                    "1" if toy in shared_toys else "0"
                )

    # Saves new splits
    for split in splits_order:
        with open(os.path.join(target_splits_directory, f"{split}.csv"), "w") as f:
            f.write(",".join(first_lines["splits"][split]) + "\n")
            id = 0
            for sequence in splits_order[split]:
                person = sequence.person
                for row_values in data[person][sequence]["values"]:
                    for view in data[person][sequence]["views"]:
                        f.write(f"{id},{sequence}/{view},{','.join(row_values)}\n")
                    id += 1

    # Creates additional splits
    if additional_splits["trainval"]:
        with open(os.path.join(target_splits_directory, f"train.csv"), "r") as f:
            train_lines = f.readlines()
        with open(os.path.join(target_splits_directory, f"validation.csv"), "r") as f:
            validation_lines = f.readlines()[1:]
        with open(os.path.join(target_splits_directory, f"trainval.csv"), "w") as f:
            f.writelines(train_lines)
            f.writelines(validation_lines)
    if additional_splits["validation_challenge"]:
        write_split_challenge("validation")
    if additional_splits["test_challenge"]:
        write_split_challenge("test")
