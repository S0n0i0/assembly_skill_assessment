# code to generate TSM-style .txt annotations from the .csv annotations provided at
# https://github.com/assembly-101/assembly101-annotations

import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# CLI arguments
# modality = ['rgb', 'mono', 'combined'] (combined = rgb+mono)
# split = ['train', 'validation', 'test', 'test_challenge']
# mode = ['recognition', 'anticipation']
modality = str(sys.argv[1])
split = str(sys.argv[2])
mode = str(sys.argv[3])

# path of the .csv annotations
anno_path = "D:/data/annotations/fine-grained-annotations"

# output path of the generated .txt
if mode == "recognition":
    out_path = "D:/data/annotations/action_recognition"
elif mode == "anticipation":
    out_path = "D:/data/annotations/action_anticipation"

# load the .csv annotations
if split == "test":
    annotations = pd.read_csv(
        f"{anno_path}/{split}.csv",
        header=0,
        names=["id", "video", "start_frame", "end_frame", "is_shared", "is_rgb"],
    )
elif split == "test_challenge" and mode == "anticipation":
    annotations = pd.read_csv(
        f"{anno_path}/{split}.csv",
        header=0,
        names=["id", "video", "start_frame", "end_frame", "is_shared"],
    )
elif split == "trainval" and mode == "anticipation":
    with open(f"{out_path}/train.csv", "r") as f:
        train_lines = f.readlines()
    with open(f"{out_path}/validation.csv", "r") as f:
        validation_lines = f.readlines()
else:
    annotations = pd.read_csv(
        f"{anno_path}/{split}.csv",
        header=0,
        names=[
            "id",
            "video",
            "start_frame",
            "end_frame",
            "action_id",
            "verb_id",
            "noun_id",
            "action_cls",
            "verb_cls",
            "noun_cls",
            "toy_id",
            "toy_name",
            "is_shared",
            "is_rgb",
        ],
    )

if split != "trainval":
    # create an empty .txt file to write into
    if mode == "prediction":
        if modality == "combined":
            file = open(f"{out_path}/{split}_combined.txt", "w+")
        elif modality == "rgb":
            file = open(f"{out_path}/{split}_rgb.txt", "w+")
        elif modality == "mono":
            file = open(f"{out_path}/{split}_mono.txt", "w+")
        else:
            print(f"Modality [{modality}] not recognized!!")
            exit()
    elif mode == "anticipation":
        file = open(f"{out_path}/{split}.csv", "w+")
        if split == "test_challenge":
            first_line = [
                "id",
                "video",
                "start_frame",
                "end_frame",
                "is_shared",
                "is_rgb",
            ]
        elif split == "test":
            first_line = [
                "id",
                "video",
                "start_frame",
                "end_frame",
                "is_shared",
                "is_rgb",
            ]
        else:
            first_line = [
                "id",
                "video",
                "start_frame",
                "end_frame",
                "action_id",
                "verb_id",
                "noun_id",
                "action_cls",
                "verb_cls",
                "noun_cls",
                "toy_id",
                "toy_name",
                "is_shared",
                "is_rgb",
            ]
        file.write(",".join(first_line) + "\n")

    last_line = None
    id = -1
    for _, a in tqdm(
        annotations.iterrows(), f"Loading annotations [{split}]", total=len(annotations)
    ):
        if split == "test_challenge":
            name, view = (a.video, "")
        else:
            name = a.video[:-4]
            name, view = name.split("/")

        modifier = 1 if mode == "recognition" else 0
        start = int(a.start_frame) + modifier
        end = int(a.end_frame) + modifier
        num_frame = end - start + modifier

        is_shared = int(a.is_shared)  # a.is_shared
        if split != "test_challenge":
            is_rgb = bool(int(a.is_rgb))  # a.is_rgb

        if mode == "anticipation":
            if split != "test" and split != "test_challenge":
                action_id = int(a.action_id if mode == "anticipation" else a.action_id)
                if last_line is None or last_line != (name, start, end, action_id):
                    last_line = (name, start, end, action_id)
                    id += 1
            elif last_line is None or last_line != (name, start, end):
                last_line = (name, start, end)
                id += 1

        # if the modality of the segment does not match the expected modality, ignore the segment
        if split != "test_challenge":
            if modality == "rgb" and not is_rgb:
                continue
            elif modality == "mono" and is_rgb:
                continue

        # write the segment annotation to the .txt
        if split == "test":
            if mode == "recognition":
                file.write(f"{name}/{view} {start} {num_frame} -1\n")
            elif mode == "anticipation":
                file.write(
                    f"{id},{name}/{view}.mp4,{start},{end},{is_shared},{int(is_rgb)}\n"
                )
        elif split == "test_challenge" and mode == "anticipation":
            file.write(f"{id},{name},{start},{end},{is_shared}\n")
        else:
            if mode == "recognition":
                file.write(f"{name}/{view} {start} {num_frame} {action_id}\n")
            elif mode == "anticipation":
                verb_id = int(a.verb_id)
                noun_id = int(a.noun_id)
                file.write(
                    f"{id},{name}/{view}.mp4,{start},{end},{action_id},{verb_id},{noun_id},{','.join([str(e) for e in a.loc['action_cls':].to_numpy().tolist()])}\n"
                )

    file.close()
else:
    with open(f"{out_path}/trainval.csv", "w") as f:
        f.writelines(train_lines)
        f.writelines(validation_lines[1:])
