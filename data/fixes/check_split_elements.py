import json

from utils.functions import all_subsets


def get_elements(lines: list[str], user_id_extractor, no_first_line):
    people = set()
    jump_line = no_first_line
    for l in lines:
        if jump_line:
            jump_line = False
            continue
        people.add(user_id_extractor(l))

    return people


def count_common_elements(common_people):
    return sum(len(common_people[common_split]) for common_split in common_people)


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            to_return = list(obj)
            to_return.sort()
            return to_return
        elif isinstance(obj, list):
            obj.sort()
            return obj
        return json.JSONEncoder.default(self, obj)


project = "recognition"  # "recognition", "anticipation", "segmentation", "fine-grained"
subject = "people"  # "people", "toys"
if project == "recognition":
    splits = ["train", "validation", "test"]
    file_format = "D:/data/annotations/action_recognition/{split}_mono.txt"
    if subject == "people":
        user_id_extractor = lambda x: x.strip().split("_")[4]
    elif subject == "toys":
        user_id_extractor = lambda x: x.strip().split("_")[3].split("-")[1]
    no_first_line = False
elif project == "anticipation":
    splits = ["train", "validation", "test"]
    file_format = "D:/data/annotations/action_anticipation/{split}.csv"
    if subject == "people":
        user_id_extractor = lambda x: x.strip().split("_")[4]
    elif subject == "toys":
        user_id_extractor = lambda x: x.strip().split("_")[3].split("-")[1]
    no_first_line = True
elif project == "segmentation":
    splits = ["train", "validation", "test"]
    file_format = "D:/data/annotations/coarse-annotations/coarse_splits/{split}_coarse_assembly.txt"
    if subject == "people":
        user_id_extractor = lambda x: x.strip().split("_")[5]
    elif subject == "toys":
        user_id_extractor = lambda x: x.strip().split("_")[4].split("-")[1]
    no_first_line = False
elif project == "fine-grained":
    splits = ["train", "validation", "test"]
    file_format = "D:/data/annotations/fine-grained-annotations/{split}.csv"
    if subject == "people":
        user_id_extractor = lambda x: x.strip().split("_")[4]
    elif subject == "toys":
        user_id_extractor = lambda x: x.strip().split("_")[3].split("-")[1]
    no_first_line = True
elif project == "debug":
    print("Debug data")
    splits = ["train", "validation", "test"]
    user_id_extractor = lambda x: x
    no_first_line = False
    data = {
        "train": ["1", "2", "1", "5"],
        "validation": ["3", "4", "5"],
        "test": ["4", "2", "5"],
    }

dump_directory = None  # "D:/"
original_elements = {split: set() for split in splits}
elements = {split: set() for split in splits}
elements["common"] = {"_".join(split_comb): set() for split_comb in all_subsets(splits)}

for split in splits:
    last_p = None
    continue_first = no_first_line

    if project != "debug":
        with open(file_format.format(split=split), "r") as f:
            lines = f.readlines()
    else:
        lines = data[split]

    original_elements[split] = get_elements(lines, user_id_extractor, no_first_line)
if dump_directory is not None:
    json.dump(
        original_elements,
        open(dump_directory + "all_people.json", "w"),
        cls=SetEncoder,
        indent=2,
    )

# Every split has a set of people who are only in that split
for split in splits:
    tmp_people_split = original_elements[split].copy()
    for s in splits:
        if s != split:
            tmp_people_split -= original_elements[s]
    elements[split] = tmp_people_split

for split_comb in elements["common"]:
    for split in split_comb.split("_"):
        elements["common"][split_comb] = (
            elements["common"][split_comb]
            if len(elements["common"][split_comb]) > 0
            else original_elements[split]
        ) & original_elements[split]
if dump_directory is not None:
    json.dump(
        elements,
        open(dump_directory + "people_division.json", "w"),
        cls=SetEncoder,
        indent=2,
    )

for split_comb in sorted(
    elements["common"], key=lambda x: len(x.split("_")), reverse=True
):
    for smaller_comb in all_subsets(split_comb.split("_"), 2, False):
        smaller_key = "_".join(smaller_comb)
        if smaller_key in elements["common"]:
            elements["common"][smaller_key] -= elements["common"][split_comb]

count = 0
while count < len(elements["common"]):
    split_comb = list(elements["common"].keys())[count]
    if len(elements["common"][split_comb]) == 0:
        del elements["common"][split_comb]
    else:
        count += 1
if dump_directory is not None:
    json.dump(
        elements,
        open(dump_directory + "people_division_diff_common.json", "w"),
        cls=SetEncoder,
        indent=2,
    )

total_common = sum(
    len(elements["common"][common_split]) for common_split in elements["common"]
)
print(f"TOTAL: {sum([len(elements[split]) for split in splits]) + total_common}")
for split in splits:
    print(f"{split}: {len(elements[split])}")
print(f"Common: {total_common}")
for common_split in elements["common"]:
    print(f"- {common_split}: {len(elements['common'][common_split])}")
