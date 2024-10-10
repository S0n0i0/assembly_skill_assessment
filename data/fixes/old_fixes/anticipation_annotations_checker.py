split = "test"
to_edit_path = f"D:/data/annotations/action_anticipation/old/{split}.csv"
ref_path = f"D:/data/annotations/action_anticipation/{split}.csv"
modes = ["clean_ids", "check_diff"]  # "clean_ids" or "check_diff"
save_edits = False
if "clean_ids" in modes:
    with open(to_edit_path, "r") as f:
        lines = f.readlines()
    with open(ref_path, "r") as f:
        ref_lines = f.readlines()

    steps = {0: 0}
    last_step = 0
    count = 2
    last_id = int(ref_lines[1].split(",")[0])
    for l in lines[2:]:
        id = int(l.split(",")[0])
        modified_id = id + steps[last_step]
        try:
            ref_id = int(ref_lines[count].split(",")[0])
            count += 1
            if modified_id != ref_id:
                if last_id != ref_id:
                    steps[id] = ref_id - id
                    last_step = id
                    print(f"Modifier for id {id}: {steps[id]} -> {ref_id}")
                else:
                    count += 1
                    while int(ref_lines[count].split(",")[0]) == last_id:
                        count += 1
        except IndexError:
            pass
        finally:
            if ref_id != last_id:
                last_id = ref_id

    to_write = [lines[0]]
    actual_step = steps[0]
    del steps[0]
    for line in lines[1:]:
        id = int(line.split(",")[0])
        if id in steps:
            actual_step = steps[id]
            del steps[id]
        id_to_write = str(id + actual_step)
        to_write.append(id_to_write + "," + ",".join(line.split(",")[1:]))

    if save_edits:
        with open(to_edit_path, "w") as f:
            f.writelines(to_write)

if "check_diff" in modes:
    if "clean_ids" in modes:
        lines = set(to_write[1:])
        lines_by_id = {int(l.split(",")[0]): l for l in lines}
    else:
        with open(to_edit_path, "r") as f:
            lines = set(f.readlines()[1:])
            lines_by_id = {int(l.split(",")[0]): l for l in lines}
    with open(ref_path, "r") as f:
        ref_lines = set(f.readlines()[1:])  # Clean multiple instances of the same row
    lines_diff = ref_lines - lines

    print("Total diffs:", len(lines_diff))
    for l in lines_diff:  # Check for further differences
        id = int(l.split(",")[0])
        if id in lines_by_id:
            name = l.split(",")[1].split("/")[0]
            values = l.split(",")[2:]
            ref_name = lines_by_id[id].split(",")[1].split("/")[0]
            ref_values = lines_by_id[id].split(",")[2:]
            if name != ref_name or values != ref_values:
                print(f"Diff for id {id}: {l} -> {lines_by_id[id]}")
