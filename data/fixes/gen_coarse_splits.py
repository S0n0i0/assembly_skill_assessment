import pandas as pd

from utils.constants import first_lines

fine_splits_directory = "D:/data/assembly/annotations/action_anticipation/"
coarse_splits_directory = (
    "D:/data/assembly/annotations/coarse-annotations/coarse_splits/"
)
splits = ["train", "validation", "test"]

for split in splits:
    sequences = {}
    lines = pd.read_csv(
        fine_splits_directory + split + ".csv",
        header=0,
        names=first_lines["splits"][split],
    )
    for _, row in lines.iterrows():
        sequence = row["video"].split("/")[0]
        if sequence not in sequences:
            if split == "validation":
                a = row["is_shared"]
            sequences[sequence] = {
                "is_shared": "shared" if row["is_shared"] == 1 else "notshared",
                "toy_id": (
                    row["toy_id"]
                    if "toy_id" in row
                    else sequence.split("_")[3].split("-")[1]
                ),
            }
        elif (
            sequences[sequence]["is_shared"] == "notshared" and row["is_shared"] == 1
        ):
            sequences[sequence]["is_shared"] = "shared"

    with open(coarse_splits_directory + split + "_coarse_assembly.txt", "w") as f:
        for sequence in sequences:
            if split == "train":
                f.write(
                    "assembly_"
                    + sequence
                    + ".txt\t\t-\t"
                    + sequences[sequence]["toy_id"]
                    + "\t-\n"
                )
            elif split == "validation":
                f.write(
                    "assembly_"
                    + sequence
                    + ".txt\t\t"
                    + sequences[sequence]["is_shared"]
                    + "\t"
                    + sequences[sequence]["toy_id"]
                    + "\n"
                )  # TODO: insert toy name (atm no id-name file provided)
            elif split == "test":
                f.write(
                    "assembly_"
                    + sequence
                    + ".txt\t\t"
                    + sequences[sequence]["is_shared"]
                    + "\n"
                )
