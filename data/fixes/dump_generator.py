import os
import random
import cv2
import pickle


def gen_object_source(dump_directory, videos_directory):
    instructions_annotations = {}
    for sequence in os.listdir(os.path.join(videos_directory)):
        instructions_annotations[sequence] = {}
        sequence_path = os.path.join(videos_directory, sequence)
        if not os.path.isdir(sequence_path):
            continue
        views = [view[:-4] for view in os.listdir(sequence_path)]
        # get random view
        view = random.choice(views)
        view_path = os.path.join(sequence_path, view + ".mp4")
        cap = cv2.VideoCapture(view_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # get resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        for i in range(frame_count):
            if random.random() < 0.01:
                # Get a random bounding box
                x = random.randint(0, width)
                y = random.randint(0, height)
                w = random.randint(0, width - x)
                h = random.randint(0, height - y)
                instructions_annotations[sequence][i] = {view: [x, y, w, h]}

    # dump the annotations in a pickle file
    with open(
        os.path.join(dump_directory, "gaze_analysis/instructions_annotations.pkl"), "wb"
    ) as f:
        pickle.dump(instructions_annotations, f)


annotations_type = ["ego"]
dump_directory = "D:/data/dumps/"
videos_directories = {a: f"D:/data/videos/{a}_recordings/" for a in annotations_type}
to_generate = [
    {
        "generator": gen_object_source,
        "args": [dump_directory, videos_directories["ego"]],
    },
]

for generator in to_generate:
    generator["generator"](*generator["args"])
