import os
from itertools import cycle
import numpy as np
from statistics import mean
from typing import TypeAlias

from utils.constants import first_lines
from utils.classes import Sequence


SplitsOrder: TypeAlias = dict[str, list[Sequence]]
SplitsOrderArchive: TypeAlias = list[
    list[dict[str, list[Sequence]]] | None,
    tuple[str, int] | None,
    tuple[str, int] | None,
]
ActualSplitsDivisions: TypeAlias = dict[str, dict[str, list[str]]]
SplitsDivisions: TypeAlias = dict[str, str]
SplitsData: TypeAlias = dict[str, dict[Sequence, dict[str, any]]]


def map_skill_level(skill_aggregations, skill_level):
    for skill_level_aggregation in skill_aggregations:
        if skill_level in skill_aggregations[skill_level_aggregation]:
            return skill_level_aggregation
    return None


def get_splits_distributions(
    splits_divisions: SplitsDivisions,
    data: SplitsData,
    log=False,
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

    if log:
        print("_Videos per split/skill level:")
        for split in people_per_split:
            print(f"- {split}: {splits_distributions[split]['skills']}")
        """print("_Duration:")
        for split in people_per_split:
            print(f"- {split}: {splits_distributions[split]['duration']}")"""

    return splits_distributions


def check_balance(splits_distributions, log=False):
    global people_per_split

    total_videos = {}
    skill_dristributions_diffs = []
    # durations = []
    for split in splits_distributions:
        # durations.append(splits_distributions[split]["duration"])
        total_videos[split] = sum(splits_distributions[split]["skills"])

    splits_percentages = {split: [] for split in people_per_split}
    for i in range(len(skill_aggregations)):
        percentage_skills_diff = []
        for split in people_per_split:
            percentage_skills_diff.append(
                splits_distributions[split]["skills"][i] / total_videos[split]
            )
            splits_percentages[split].append(round(percentage_skills_diff[-1] * 100, 2))
        skill_dristributions_diffs.append(
            max(percentage_skills_diff) - min(percentage_skills_diff)
        )
        """skill_dristributions_diffs.append(
            max(splits_distributions[split]["skills"])
            - min(splits_distributions[split]["skills"])
            < balance_tollerances["skills"]
        )"""

    if log:
        print("_Percentages:")
        for split in splits_percentages:
            print(f"- {split}: {tuple(splits_percentages[split])}")
    return {
        "processed_info": {
            "percentage_skills_diff": {
                "percentages": splits_percentages,
                "diffs": skill_dristributions_diffs,
            },
            # "duration": durations,
        },
        "balance": {
            # "skills": all(skill_dristributions_diffs),
            "percentage_skills_diff": all(
                [
                    diff < balance_tollerances["percentage_skills_diff"]
                    for diff in skill_dristributions_diffs
                ]
            ),
            # "duration": max(durations) - min(durations) < balance_tollerances["duration"],
        },
    }


def get_nearest_people(
    splits_divisions: SplitsDivisions,
    data: SplitsData,
    to_fix: tuple[str, int],
    target: tuple[str, int],
    splits_order_archive: SplitsOrderArchive | None = None,
):
    global people_per_split

    get_skill_mean = lambda person: mean(
        [data[person][sequence]["skill_level"] for sequence in data[person]]
    )

    to_fix_candidates = [
        (person, get_skill_mean(person))
        for person in splits_divisions.keys()
        if splits_divisions[person] == to_fix[0]
    ]
    to_fix_candidates.sort(key=lambda x: abs(x[1] - to_fix[1]))
    target_candidates = [
        (person, get_skill_mean(person))
        for person in splits_divisions.keys()
        if splits_divisions[person] == target[0]
    ]
    target_candidates.sort(key=lambda x: abs(x[1] - target[1]))

    i = 0
    j = 0
    if splits_order_archive is not None:
        again = True
        while again and i < len(to_fix_candidates) and j < len(target_candidates):
            if is_old_movement(
                splits_order_archive, to_fix_candidates[i][1], target_candidates[j][1]
            ):
                impossible_diff = max(skill_aggregations.keys()) + 1
                next_to_fix_diff = (
                    abs(to_fix_candidates[i + 1][1] - to_fix[1])
                    if i + 1 < len(to_fix_candidates)
                    else impossible_diff
                )
                next_target_diff = (
                    abs(target_candidates[j + 1][1] - target[1])
                    if j + 1 < len(target_candidates)
                    else impossible_diff
                )
                if next_to_fix_diff < next_target_diff:
                    i += 1
                elif next_to_fix_diff > next_target_diff:
                    j += 1
                elif (
                    next_to_fix_diff == impossible_diff
                    and next_target_diff == impossible_diff
                ):
                    i = -1
                    j = -1
                    again = False
                else:
                    i += 1  # Possibility to add further conditions
            else:
                again = False

    return (
        (to_fix_candidates[i][0], target_candidates[j][0])
        if i > -1 and j > -1
        else (None, None)
    )


def move_person(
    splits_order: SplitsOrder,
    splits_divisions: SplitsDivisions,
    data: SplitsData,
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
    global target_splits_directory, first_lines, data, splits_order

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


def get_last_splits_order(splits_order_archive: SplitsOrderArchive):
    for step in reversed(splits_order_archive):
        if step[0] is not None:
            return step[0]

    return None


def is_old_movement(
    splits_order_archive: SplitsOrderArchive, to_fix_level: float, target_level: float
):
    k = len(splits_order_archive) - 1
    found = False
    while not found and k >= 0:
        a = splits_order_archive[k][1][1]
        b = splits_order_archive[k][2][1]
        if to_fix_level != a and target_level != b:
            k -= 1
        else:
            c = get_last_splits_order(splits_order_archive[:k])
            d = get_last_splits_order(splits_order_archive)
            if c == d:
                found = True
            else:
                k -= 1

    return found


def get_values(values: list[str], split: str):
    return values[:2] + values[-2:] if split == "test" else values


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
    "percentage_skills_diff": 0.09,  # Difference between the percentage of a skill level in a split and the others
    # "duration": 20,  # Seconds
}

# Create a dictionary of the levels for all people
data: SplitsData = {}
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

actual_splits_order: SplitsOrder = {}
actual_splits_divisions: ActualSplitsDivisions = {}
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
            data[person][sequence]["values"].append(
                values
            )  # TODO: sistemare se values arrivano da test (non ci sono tutte le informazioni)

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

# Sort in descending order of people by total duration of their videos
people = list(data.keys())
people.sort(key=lambda p: sum([data[p][s]["end_frame"] for s in data[p]]), reverse=True)

# Place people in order to follow the people_per_split
splits_order: SplitsOrder = {split: [] for split in people_per_split}
splits_counters: dict[str, int] = {split: 0 for split in people_per_split}
splits_divisions: SplitsDivisions = {person: None for person in people}
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
repeat = False
splits_order_archive: SplitsOrderArchive = []
while again:
    print("Distribution n.", len(splits_order_archive))
    splits_order_archive.append([splits_order if not repeat else None, None, None])

    splits_distribution = get_splits_distributions(splits_divisions, data, True)
    balance_analysis = check_balance(splits_distribution, True)
    print()
    if not repeat:
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
                        splits_order = get_last_splits_order(splits_order_archive)
                        splits_order_archive = splits_order_archive[:answer]
                        repeat = True
                        right = True
                    else:
                        print("Invalid number.")
                else:
                    print("Invalid input.")
            continue

    skill_level_to_fix = (
        np.argmax(balance_analysis["processed_info"]["percentage_skills_diff"]["diffs"])
        + 1
    )
    skill_level_target = (
        np.argmin(balance_analysis["processed_info"]["percentage_skills_diff"]["diffs"])
        + 1
    )

    skills_split = {
        split: balance_analysis["processed_info"]["percentage_skills_diff"][
            "percentages"
        ][split][skill_level_to_fix - 1]
        for split in splits_distribution
    }
    split_to_fix = max(skills_split, key=skills_split.get)
    skills_split = {
        split: balance_analysis["processed_info"]["percentage_skills_diff"][
            "percentages"
        ][split][skill_level_target - 1]
        for split in splits_distribution
        if split != split_to_fix
    }
    split_target = min(skills_split, key=skills_split.get)
    splits_order_archive[-1][1] = (split_to_fix, skill_level_to_fix)
    splits_order_archive[-1][2] = (split_target, skill_level_target)

    to_fix_person, target_person = get_nearest_people(
        splits_divisions,
        data,
        (split_to_fix, skill_level_to_fix),
        (split_target, skill_level_target),
        splits_order_archive if repeat else None,
    )
    repeat = False

    if to_fix_person is None and target_person is None:
        print("No more people to move.")
        again = False
        continue

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
                        f.write(
                            f"{id},{sequence}/{view},{','.join(get_values(row_values, split))}\n"
                        )
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
