def show_groups(lines: list[str], split: str):

    last_line = (
        get_split_fn(split, "get_id")(lines[0]),
        get_split_fn(split, "get_name")(lines[0]),
        ",".join(get_split_fn(split, "get_values")(lines[0])),
    )
    possible_values = dict()
    group_count = 1
    for line in lines[1:]:
        id = get_split_fn(split, "get_id")(line)
        name = get_split_fn(split, "get_name")(line)
        values = ",".join(get_split_fn(split, "get_values")(line))

        if name == last_line[1] and values == last_line[2]:
            group_count += 1
        else:
            if group_count not in possible_values:
                possible_values[group_count] = []
            possible_values[group_count].append(last_line[0])
            last_line = (id, name, values)
            group_count = 1

    keys = list(possible_values.keys())
    keys.sort()
    for key in keys:
        print(
            f"{key}: {len(possible_values[key])} groups (es. {possible_values[key][0]} id)"
        )
    print("Total groups:", sum([len(possible_values[key]) for key in keys]))


def get_split_fn(split: str, key: str):
    return (
        utility_fns[key][split]
        if isinstance(utility_fns[key], dict)
        else utility_fns[key]
    )


mode = "coarse"
if mode == "fine":
    fine_directory = "D:/data/annotations/fine-grained-annotations/"
    splits = ["train", "validation", "test"]
    first_row_count = 1
    check_groups = True
    utility_fns = {
        "format_id": {
            "train": lambda x: f"{x:07}",
            "validation": lambda x: f"{x:07}",
            "test": lambda x: f" {x}",
        },
        "get_id": lambda x: x.split(",")[0],
        "get_name": lambda x: x.split(",")[1].split("/")[0],
        "get_values": lambda x: x.split(",")[2:5],
        "get_no_id": lambda x: x.split(",")[1:],
        "get_file": lambda x: fine_directory + x + ".csv",
        "is_ok": lambda x: "HMC_" in x,
    }
elif mode == "coarse":
    fine_directory = "D:/data/annotations/coarse-annotations/"
    splits = ["coarse_seq_views"]
    first_row_count = 0
    check_groups = False
    utility_fns = {
        "format_id": None,
        "get_no_id": lambda x: [x],
        "get_file": lambda x: fine_directory + x + ".txt",
        "is_ok": lambda x: "HMC_" in x and x[:9] == "assembly_",
    }
for split in splits:
    print("\nSplit:", split)
    file = get_split_fn(split, "get_file")(split)
    with open(file) as f:
        lines = f.readlines()

    if check_groups:
        print("[Before] Groups sizes:")
        show_groups(lines[1:], split)

    with open(file, "w+") as f:
        if first_row_count == 1:
            f.write(lines[0])
        count = 0
        for line in lines[first_row_count:]:
            if get_split_fn(split, "is_ok")(line):
                to_write = get_split_fn(split, "get_no_id")(line)
                if utility_fns["format_id"] is not None:
                    to_write = get_split_fn(split, "format_id")(count) + to_write

                f.write(",".join(to_write))
                count += 1

        if check_groups:
            f.seek(0)
            print("[After] Groups sizes:")
            show_groups(f.readlines(), split)
