import os

original_actions_path = (
    "D:/data/annotations/fine-grained-annotations/tmp/tmp/actions.csv"
)
original_splits_path = "D:/data/annotations/fine-grained-annotations/tmp/"
new_actions_path = "D:/data/annotations/fine-grained-annotations/actions.csv"
target_path = "D:/data/annotations/fine-grained-annotations/"

with open(original_actions_path, "r") as original_actions_file:
    original_actions = original_actions_file.readlines()[1:]
    original_actions = {
        line.split(",")[1]: line.split(",")[4] for line in original_actions
    }
with open(new_actions_path, "r") as new_actions_file:
    new_actions = new_actions_file.readlines()[1:]
    new_actions = {
        line.split(",")[4]: [f"{int(code):04}" for code in line.split(",")[1:4]]
        for line in new_actions
    }

for split in os.listdir(original_splits_path):
    split_path = os.path.join(original_splits_path, split)
    if os.path.isfile(split_path):
        with open(split_path, "r") as split_file:
            split_lines = split_file.readlines()
        new_split_lines = [split_lines[0]]
        skipped_actions = {}
        count = 0
        for line in split_lines[1:]:
            line = line.split(",")
            code = str(int(line[4]))
            if code in original_actions and original_actions[code] in new_actions:
                if split == "train.csv" or split == "validation.csv":
                    new_line = (
                        [f"{count:07}"]
                        + line[1:4]
                        + new_actions[original_actions[code]]
                        + line[7:]
                    )
                else:
                    new_line = [f"{count:07}"] + line[1:]
                new_split_lines.append(",".join(new_line))
                count += 1
            else:
                if code not in skipped_actions:
                    skipped_actions[code] = set()
                skipped_actions[code].add(original_actions[code])

        target_split_path = os.path.join(target_path, split)
        with open(target_split_path, "w") as target_split_file:
            target_split_file.writelines(new_split_lines)
            print(f"\nSaved {len(new_split_lines)} lines to {target_split_path}")
        if len(skipped_actions) > 0:
            print(
                f"Not mapped (exluded: 'attempt' and 'inspect' actions): {len(skipped_actions)}"
            )
            print(
                [
                    f"{k}: {v}"
                    for k, v in skipped_actions.items()
                    if "attempt" in v or "inspect" in v
                ]
            )
