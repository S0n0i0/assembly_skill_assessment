with open("D:/data/annotations/action_anticipation/train.csv") as anticipation_file:
    a_lines = anticipation_file.readlines()
    with open(
        "D:/data/annotations/coarse-annotations/coarse_splits/train_coarse_assembly.txt"
    ) as segmentation_file:
        s_lines = segmentation_file.readlines()
        s_lines = [s_line.split("	")[0] for s_line in s_lines]
        for s_line in s_lines:
            dir_name = "_".join(s_line.split("		")[0].split(".")[0].split("_")[1:])
            a_frames = list(
                set(
                    [
                        int(a_line.split(",")[2])
                        for a_line in a_lines[1:]
                        if a_line.split(",")[1].split("/")[0] == dir_name
                    ]
                )
            )
            a_frames.sort()
            with open(
                f"D:/data/annotations/coarse-annotations/coarse_labels/{s_line}"
            ) as s_label_file:
                sl_lines = s_label_file.readlines()
                equal = True
                frame = int(sl_lines[0].split("	")[0])
                if frame not in a_frames and (frame + 1) not in a_frames:
                    equal = False
                    print("First:", frame)
                frame = int(sl_lines[-1].split("	")[1])
                if equal:
                    if frame not in a_frames and (frame - 1) not in a_frames:
                        equal = False
                        print("Last:", frame)
                if equal:
                    sl_frames = [
                        int(sl_line.split("	")[1]) for sl_line in sl_lines[:-1]
                    ]
                    for frame in sl_frames:
                        if frame not in a_frames:
                            print("Middle", frame)
                            equal = False
                            break
                if not equal:
                    print(f"Problem with: {s_line}")
