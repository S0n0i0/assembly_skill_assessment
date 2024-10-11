import os

from utils.classes import Sequence

sequences_directory = "D:/data/ego_recordings/"
fine_directory = "D:/data/annotations/action_anticipation/"
splits = ["train", "validation", "validation_challenge", "test", "test_challenge"]

sequences = set(
    [
        d
        for d in os.listdir(sequences_directory)
        if os.path.isdir(os.path.join(sequences_directory, d))
    ]
)
total_peoples = set([Sequence(d).person for d in sequences])

for split in splits:
    with open(fine_directory + f"{split}.csv", "r") as f:
        lines = set([line.split(",")[1] for line in f.readlines()])

    sequences -= lines


print("Videos absent from annotations:", len(sequences))
peoples = set([Sequence(d).person for d in sequences])
missing_peoples = total_peoples - peoples
print(
    "People with no videos in annotations:",
    len(missing_peoples),
    "out of",
    len(total_peoples),
)
