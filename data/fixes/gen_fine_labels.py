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
# split = ['train', 'validation', 'test']
modality = str(sys.argv[1])
split = str(sys.argv[2])

# path of the .csv annotations
anno_path = "D:/data/annotations/action_anticipation"

# output path of the generated .txt
out_path = "D:/data/annotations/action_recognition"

# load the .csv annotations
if split == "test":
    annotations = pd.read_csv(
        f"{anno_path}/{split}.csv",
        header=0,
        names=["id", "video", "start_frame", "end_frame", "is_shared", "is_rgb"],
    )
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
    if modality == "combined":
        file = open(f"{out_path}/{split}_combined.txt", "w+")
    elif modality == "rgb":
        file = open(f"{out_path}/{split}_rgb.txt", "w+")
    elif modality == "mono":
        file = open(f"{out_path}/{split}_mono.txt", "w+")
    else:
        print(f"Modality [{modality}] not recognized!!")
        exit()

    last_line = None
    id = -1
    for _, a in tqdm(
        annotations.iterrows(), f"Loading annotations [{split}]", total=len(annotations)
    ):
        name = a.video[:-4]
        name, view = name.split("/")

        modifier = 1
        start = int(a.start_frame) + modifier
        end = int(a.end_frame) + modifier
        num_frame = end - start + modifier

        if split != "test":
            action_id = int(a.action_id)

        is_rgb = bool(int(a.is_rgb))  # a.is_rgb

        # if the modality of the segment does not match the expected modality, ignore the segment
        if modality == "rgb" and not is_rgb:
            continue
        elif modality == "mono" and is_rgb:
            continue

        # write the segment annotation to the .txt
        if split == "test":
            file.write(f"{name}/{view} {start} {num_frame} -1\n")
        else:
            file.write(f"{name}/{view} {start} {num_frame} {action_id}\n")

    file.close()
