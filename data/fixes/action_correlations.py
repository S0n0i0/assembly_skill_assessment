import os
import numpy as np

splits = ["train", "validation", "test"]
fine_labels_path = "D:/data/assembly/annotations/action_anticipation/"
coarse_labels_path = "D:/data/assembly/annotations/coarse-annotations/coarse_labels/"
recordings_path = "D:/data/assembly/videos/ego_recordings/"
annotations_fps = 30


fine_mean = 0
fine_ranges = {}
for split in splits:
    with open(fine_labels_path + split + ".csv") as fine_labels_file:
        previous_range = []
        split_mean = 0
        split_actions = 0
        for line in fine_labels_file.readlines()[1:]:
            divided_line = line.split(",")
            sequence = divided_line[1].split("/")[0]
            if sequence not in fine_ranges:
                fine_ranges[sequence] = []
                previous_range = []
            fine_range = list(map(int, divided_line[2:4]))
            if fine_range != previous_range:
                fine_ranges[sequence].append(fine_range)
                split_mean += fine_range[1] - fine_range[0]
                split_actions += 1
                previous_range = fine_range
        fine_mean += split_mean / split_actions

fine_mean /= len(splits)
print("Mean fine label:", fine_mean / annotations_fps)

sequences = [
    d
    for d in os.listdir(recordings_path)
    if os.path.isdir(os.path.join(recordings_path, d))
]
coarse_mean = 0
coarse_ranges = {}
for sequence in sequences:
    with open(
        coarse_labels_path + "assembly_" + sequence + ".txt"
    ) as coarse_labels_file:
        if sequence not in coarse_ranges:
            coarse_ranges[sequence] = []
        sequence_mean = 0
        lines = coarse_labels_file.readlines()
        for line in lines:
            coarse_range = list(map(int, line.split("\t")[0:2]))
            coarse_ranges[sequence].append(coarse_range)
            sequence_mean += coarse_range[1] - coarse_range[0]
        coarse_mean += sequence_mean / len(lines)

coarse_mean /= len(sequences)
print("Average duration of coarse actions:", coarse_mean / annotations_fps)
print("Coarse actions distribution:")
coarse_actions_number = [len(e) for e in coarse_ranges.values()]
print("- Max number of action in a sequence:", max(coarse_actions_number))
print("- Min number of action in a sequence:", min(coarse_actions_number))
print("- Average number of action in a sequence:", np.mean(coarse_actions_number))

# Check if, for every coarse range, there are fine ranges that are contained in it and there aren't any fine ranges that are partially contained in it
not_contained_ranges = {}
for sequence in sequences:
    for i, coarse_range in enumerate(coarse_ranges[sequence]):
        not_contained_ranges[i] = []
        for fine_range in fine_ranges[sequence]:
            if (
                fine_range[0] < coarse_range[0]
                and fine_range[1] > coarse_range[0]
                or fine_range[0] < coarse_range[1]
                and fine_range[1] > coarse_range[1]
            ):
                not_contained_ranges[i].append((fine_range, coarse_range))
                break
# coarse   |     |
# fine    | |   | |

actual_not_contained_ranges = {
    i: e for i, e in not_contained_ranges.items() if len(e) > 0
}
print(
    "Coarse actions which have ranges of fine actions not entirely contained:",
    len(actual_not_contained_ranges),
    "/",
    len(not_contained_ranges),
)
print(
    "Mean not contained ranges:",
    np.mean([len(e) for e in actual_not_contained_ranges.values()]),
)
print(actual_not_contained_ranges)