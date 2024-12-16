import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json
import sys

from utils.classes import Sequence

# Global variables for buttons
bnext = None
bprev = None


# Function to plot the graphs on the current page
def plot_page(page):
    global data, bnext, bprev, graphs_per_page, num_pages, graphs_grid

    plt.clf()
    start_index = page * graphs_per_page
    end_index = min(start_index + graphs_per_page, len(data))

    keys = list(data.keys())

    for i in range(start_index, end_index):
        plt.subplot(graphs_grid[0], graphs_grid[1], i - start_index + 1)
        x_values = data[keys[i]][0][1]
        y_values = data[keys[i]][1][1]
        plt.bar(x_values, y_values)
        # Change axis names
        plt.xlabel(data[keys[i]][0][0])
        plt.ylabel(data[keys[i]][1][0])
        plt.title(keys[i])

    bnext.ax.set_visible(page < num_pages - 1)
    bprev.ax.set_visible(page > 0)

    # add space between subplots not overlapping buttons
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    plt.draw()


# Callback function for the next button
def next_page(event):
    global current_page
    if current_page < num_pages - 1:
        current_page += 1
        plot_page(current_page)


# Callback function for the previous button
def prev_page(event):
    global current_page, data
    if current_page > 0:
        current_page -= 1
        plot_page(current_page)


splits = ["train", "validation", "test"]
use_joint = True
directory_paths = {
    "coarse": "D:/data/annotations/coarse-annotations/",
    "skill": "D:/data/annotations/skill_evaluation/",
}
labels_directories = {
    "coarse": os.path.join(directory_paths["coarse"], "coarse_labels"),
    "skill": os.path.join(directory_paths["skill"], "skill_labels"),
}
skill_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
skill_levels = list(set(skill_mapping.values()))
plots = {
    "coarse_actions_distribution": True,
    "coarse_actions_distribution_per_people": True,
    "person_skills": True,
    "skill_samples": True,
    "skill_samples_per_people": True,
}

joint_actions = {}
if use_joint:
    with open(os.path.join(directory_paths["coarse"], "joint_labels.csv")) as f:
        f.readline()
        for line in f:
            _, sequence, start_frame, end_frame = line.strip().split(",")
            actual_sequence = Sequence(sequence)
            if actual_sequence not in joint_actions:
                joint_actions[actual_sequence] = []
            joint_actions[actual_sequence].append(
                [int(start_frame), int(end_frame), False]
            )

attempt_count = 0
total_actions = 0
durations = []
actions_per_file = []

sequences = os.listdir(labels_directories["coarse"])
for sequence in sequences:
    if sequence.endswith(".txt"):
        with open(os.path.join(labels_directories["coarse"], sequence), "r") as file:
            sequence = Sequence(
                sequence[:-4].replace("disassembly_", "").replace("assembly_", "")
            )
            if sequence in joint_actions:
                sequence_present = True
            else:
                sequence_present = False
            tmp_actions_per_file = 0
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    if sequence_present:
                        found = False
                        actual_start = int(parts[0])
                        for i, action_list in enumerate(joint_actions[sequence]):
                            if (
                                actual_start >= action_list[0]
                                and actual_start < action_list[1]
                            ):
                                found = True
                                break
                        if found and not action_list[2]:
                            duration = (action_list[1] - action_list[0]) / 30
                            durations.append(
                                [
                                    sequence,
                                    parts[2],
                                    duration,
                                    parts[0],
                                    sequence.person,
                                ]
                            )
                            joint_actions[sequence][i][2] = True
                        elif not found:
                            duration = (float(parts[1]) - float(parts[0])) / 30
                            durations.append(
                                [
                                    sequence,
                                    parts[2],
                                    duration,
                                    parts[0],
                                    sequence.person,
                                ]
                            )
                    else:
                        duration = (float(parts[1]) - float(parts[0])) / 30
                        durations.append(
                            [sequence, parts[2], duration, parts[0], sequence.person]
                        )
                    total_actions += 1
                    tmp_actions_per_file += 1
                    if "attempt" in parts[2]:
                        attempt_count += 1
            actions_per_file.append([sequence, tmp_actions_per_file])

print()
print(f"Total number of actions containing the word 'attempt': {attempt_count}")
print(f"Total number of actions: {total_actions}")
num_files = len(
    [name for name in os.listdir(labels_directories["coarse"]) if name.endswith(".txt")]
)
mean_attempts_per_file = attempt_count / num_files if num_files > 0 else 0
print(f"Mean number of attempts per file: {mean_attempts_per_file:.2f}")
print(f"Mean number of actions per file: {total_actions / num_files:.2f}")
durations.sort(key=lambda x: x[2])
print(f"Top 5 min actions duration: {durations[:5]}")
print(f"Top 5 max action duration: {list(reversed(durations[-5:]))}")
actions_per_file.sort(key=lambda x: x[1])
print(f"Top 5 min actions per file: {actions_per_file[:5]}")
print(f"Top 5 max actions per file: {list(reversed(actions_per_file[-5:]))}")

skill_labels = {}
for skill_label in os.listdir(labels_directories["skill"]):
    skill_level = skill_mapping[int(skill_label[:-4].split("_")[1])]
    with open(os.path.join(labels_directories["skill"], skill_label)) as f:
        for line in f:
            sequence = Sequence(line.strip())
            if sequence.person not in skill_labels:
                skill_labels[sequence.person] = {}
            skill_labels[sequence.person][sequence] = skill_level

# Create a dictionary person-durations from durations
person_durations = {}
person_skills = {}
for i, (sequence, action, duration, start, person) in enumerate(durations):
    if person not in person_durations:
        person_durations[person] = [
            ("Sample", []),
            ("Duration", []),
        ]
    if person not in person_skills:
        person_skills[person] = [
            ("Skill Level", skill_levels),
            ("Count", [0] * len(skill_levels)),
        ]

    person_durations[person][0][1].append(len(person_durations[person][0][1]) + 1)
    person_durations[person][1][1].append(duration)

    person_skills[person][1][1][skill_labels[person][sequence] - 1] += 1

for person in person_durations:
    person_durations[person][1][1].sort()

skill_samples = {
    split: [
        ("Skill Level", skill_levels),
        ("Count", [0] * len(skill_levels)),
    ]
    for split in splits
}
person_skill_samples = {}
print("Skill samples:")
skills_per_split = []
mean_skill = {split: 0 for split in skill_samples}
min_skill = {split: sys.maxsize for split in skill_samples}
max_skill = {split: -1 for split in skill_samples}
for i, split in enumerate(skill_samples):
    with open(
        os.path.join(directory_paths["skill"], f"skill_splits/{split}.csv")
    ) as split_file:
        lines = split_file.readlines()[1:]
    for line in lines:
        divided_line = line.strip().split(",")
        sequence = Sequence(divided_line[1])
        person = sequence.person
        skill_level = int(divided_line[-1])

        if person not in person_skill_samples:
            person_skill_samples[person] = [
                ("Skill Level", skill_levels),
                ("Count", [0] * len(skill_levels)),
            ]

        skill_samples[split][1][1][skill_level - 1] += 1
        person_skill_samples[person][1][1][skill_level - 1] += 1

    for person in person_skill_samples:
        samples = sum(person_skill_samples[person][1][1])
        mean_skill[split] += samples
        min_skill[split] = min(min_skill[split], samples)
        max_skill[split] = max(max_skill[split], samples)
    mean_skill[split] /= len(person_skill_samples)

    skills_per_split.append(sum(skill_samples[split][1][1]))
    print(f"_{split}: {skills_per_split[-1]}")
    print(f"- Mean: {mean_skill[split]}")
    print(f"- Min: {min_skill[split]}")
    print(f"- Max: {max_skill[split]}")
print("Total:", sum(skills_per_split))

if plots["coarse_actions_distribution"]:
    actual_durations = [x[2] for x in durations]
    plt.hist(actual_durations, bins=50, edgecolor="black")

    plt.ylabel("Coarse actions")
    plt.xlabel("Durations")
    plt.xticks(
        [
            0,
            5,
        ]
        + list(range(10, 120, 10))
    )

    plt.show()

if plots["coarse_actions_distribution_per_people"]:
    # Setup People-Durations plot
    plt.figure(figsize=(14, 9))
    plt.suptitle("People-Durations", fontsize=16)
    plt.subplots_adjust(bottom=0.2)

    # Add buttons for navigation
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next")
    bnext.on_clicked(next_page)
    bprev = Button(axprev, "Previous")
    bprev.on_clicked(prev_page)

    # Preare person durations plot
    data = person_durations
    # Number of graphs per page
    graphs_grid = (3, 3)
    graphs_per_page = graphs_grid[0] * graphs_grid[1]
    num_pages = (len(data) + graphs_per_page - 1) // graphs_per_page
    # Current page index
    current_page = 0

    plot_page(current_page)

    plt.show()

if plots["person_skills"]:
    # Setup people-skills plot
    plt.figure(figsize=(14, 9))
    plt.suptitle("People-CoarseSkills", fontsize=16)
    plt.subplots_adjust(bottom=0.2)

    # Add buttons for navigation
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next")
    bnext.on_clicked(next_page)
    bprev = Button(axprev, "Previous")
    bprev.on_clicked(prev_page)

    # Prepare person skills plot
    data = person_skills
    # Number of graphs per page
    graphs_grid = (3, 3)
    graphs_per_page = graphs_grid[0] * graphs_grid[1]
    num_pages = (len(data) + graphs_per_page - 1) // graphs_per_page
    # Current page index
    current_page = 0

    plot_page(current_page)

    plt.show()

if plots["skill_samples"]:
    # Plot bar chart for skill samples
    plt.figure(figsize=(10, 4))
    for i, split in enumerate(skill_samples):
        plt.subplot(1, 3, i + 1)
        plt.bar(skill_samples[split][0][1], skill_samples[split][1][1])
        plt.xlabel(skill_samples[split][0][0])
        plt.ylabel(skill_samples[split][1][0])
        plt.title(split)
    plt.title("Skill Samples")
    plt.tight_layout()

    plt.show()

if plots["skill_samples_per_people"]:
    # Setup people-skills plot
    plt.figure(figsize=(14, 9))
    plt.suptitle("People-GroupedSkills", fontsize=16)
    plt.subplots_adjust(bottom=0.2)

    # Add buttons for navigation
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next")
    bnext.on_clicked(next_page)
    bprev = Button(axprev, "Previous")
    bprev.on_clicked(prev_page)

    # Prepare person skills plot
    data = person_skill_samples
    # Number of graphs per page
    graphs_grid = (3, 3)
    graphs_per_page = graphs_grid[0] * graphs_grid[1]
    num_pages = (len(data) + graphs_per_page - 1) // graphs_per_page
    # Current page index
    current_page = 0

    plot_page(current_page)

    plt.show()


with open(
    os.path.join(
        directory_paths["coarse"],
        f"action_durations{'_with_joints' if use_joint else ''}.json",
    ),
    "w",
) as f:
    for i in person_durations:
        person_durations[i] = person_durations[i][1][1]
    json.dump(person_durations, f, indent=4)
with open(
    os.path.join(
        directory_paths["skill"],
        f"coarse_skills_distribution{'_with_joints' if use_joint else ''}.json",
    ),
    "w",
) as f:
    for i in person_skills:
        person_skills[i] = person_skills[i][1][1]
    json.dump(person_skills, f, indent=4)
with open(
    os.path.join(
        directory_paths["skill"],
        f"grouped_skills_distribution{'_with_joints' if use_joint else ''}.json",
    ),
    "w",
) as f:
    for i in person_skill_samples:
        person_skill_samples[i] = person_skill_samples[i][1][1]
    json.dump(person_skill_samples, f, indent=4)
