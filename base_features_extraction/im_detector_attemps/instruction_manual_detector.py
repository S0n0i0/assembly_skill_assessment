import cv2
import numpy as np
import os
from utils.classes import VideoReader


def preprocess_image(image, mask, threshold):
    if mask is not None:
        actual_image = apply_mask(image, mask)
    else:
        actual_image = image

    manage_image(actual_image, "binary_image")
    _, binary_image = cv2.threshold(actual_image, threshold, 255, cv2.THRESH_BINARY_INV)
    manage_image(binary_image, "binary_image")
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    return binary_image


# 127 è il valore di soglia; può essere regolato
# 170 D:/data/recordings/nusar-2021_action_both_9015-c03a_9015_user_id_2021-02-02_163503/HMC_21179183_mono10bit.mp4
# 193 D:/data/recordings/nusar-2021_action_both_9021-a14_9021_user_id_2021-02-03_100733/HMC_21179183_mono10bit.mp4
# 195 D:/data/recordings/nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84358933_mono10bit.mp4
# 195 D:/data/recordings/nusar-2021_action_both_9013-c03b_9013_user_id_2021-02-24_113410/HMC_84358933_mono10bit.mp4
# 200 D:/data/recordings/nusar-2021_action_both_9021-a29_9021_user_id_2021-02-23_094113/HMC_84358933_mono10bit.mp4
# 200 D:/data/recordings/nusar-2021_action_both_9021-b05c_9021_user_id_2021-02-23_094649/HMC_84358933_mono10bit.mp4
# 215 D:/data/recordings/nusar-2021_action_both_9013-b01a_9013_user_id_2021-02-02_135446/HMC_21110305_mono10bit.mp4
# 215 D:/data/recordings/nusar-2021_action_both_9016-a24_9016_user_id_2021-02-17_135905/HMC_84358933_mono10bit.mp4
def get_bboxes(image, mask, threshold, min_area=30, max_area=500):
    binary_image = preprocess_image(image, mask, threshold)
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            filtered_contours.append(contour)

    return filtered_contours, output_image  # """
    """_, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    manage_image(binary_image, "binary_image")

    # Trova i contorni degli oggetti
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Crea un'immagine a colori per disegnare le bounding box
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Disegna le bounding box attorno agli oggetti
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return contours, output_image  # """


def get_bb(image, mask, min_area=20, max_area=400):
    binary = apply_mask(image, mask)
    binary = cv2.adaptiveThreshold(
        binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 60
    )

    # Trova i contorni degli oggetti bianchi
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    filtered_contours = []
    print()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            print(area)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            filtered_contours.append(contour)

    return filtered_contours, output_image  # """


def get_white_area(image):
    _, threshold = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)

    mask = np.zeros_like(image)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    return area, mask


def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def compare_images(folder_path):

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    max_area = 0
    max_mask = None
    max_video = None
    # Iterate over each file in the folder
    for file in files:
        # Check if the file is a video file (you can add more video file extensions if needed)
        if file.endswith(".mp4"):
            # Construct the full file path
            file_path = os.path.join(folder_path, file)
            video = VideoReader(file_path)
            tmp_area, tmp_mask = get_white_area(video.get_frame())

            if tmp_area is not None and tmp_area > max_area:
                max_area = tmp_area
                max_mask = tmp_mask
                max_video = file_path

    return max_video, max_mask


def manage_image(image, title="image", save=False):
    if save:
        cv2.imwrite(f"data/frames_examples/{title}.jpg", image)
        key = None
    else:
        cv2.imshow(title, image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return key


if __name__ == "__main__":
    mode = 4
    if mode == 0:
        # Carica l'immagine in bianco e nero
        image = cv2.imread("data/frames_examples/frame_0.jpg", cv2.IMREAD_GRAYSCALE)

        _, mask = get_white_area(image)
        bboxes, image = get_bboxes(image, mask, 195)
        manage_image(image)
    elif mode == 1:
        video_path, _ = compare_images(
            "D:/data/recordings/nusar-2021_action_both_9021-b05c_9021_user_id_2021-02-23_094649"
        )
        video = VideoReader(video_path)
        manage_image(video.get_frame())
    elif mode == 2:
        video_path = "D:/data/recordings/nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84358933_mono10bit.mp4"
        video = VideoReader(video_path, 2)
        frame = video.get_frame()

        bboxes, image = get_bboxes(frame, None, 200)
        manage_image(image, "frame")
    elif mode == 3:
        d_path = "D:/data/recordings/nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724"
        gaze_video, mask = compare_images(d_path)
        video = VideoReader(gaze_video)
        frame = video.get_frame()
        bboxes = []
        count = 0
        while frame is not None and len(bboxes) == 0:
            bboxes, image = get_bb(video.get_frame(), mask)
            count += 1
            """base_threshold = 235
            bboxes, image = get_bboxes(frame, mask, base_threshold)
            while len(bboxes) == 0 and base_threshold > 0:
                bboxes, image = get_bboxes(video.get_frame(), mask, base_threshold)
                if len(bboxes) == 0:
                    base_threshold -= 5 #"""

        manage_image(image, gaze_video)
    elif mode == 4:
        notable_videos_path = "data/notable_videos.txt"
        directory_path = "D:/data/recordings"
        video_index = -1
        # thresholds = [170, 193, 195, 200, 215]
        base_threshold = 185

        directories = os.listdir(directory_path)
        for i, d in enumerate(directories):

            if i < video_index:
                continue
            d_path = os.path.join(directory_path, d)
            if os.path.isdir(d_path):
                gaze_video, mask = compare_images(d_path)
                video = VideoReader(gaze_video)

                if video.video is None:
                    continue

                frame = video.get_frame()
                bboxes = []
                while frame is not None and len(bboxes) == 0:
                    bboxes, image = get_bb(video.get_frame(), mask)
                """bboxes, image = get_bboxes(video.get_frame(), mask, base_threshold)
                while len(bboxes) <= 1 and base_threshold > 0:
                    bboxes, image = get_bboxes(video.get_frame(), mask, base_threshold)
                    if len(bboxes) <= 1:
                        base_threshold -= 5#"""
                key = manage_image(image, gaze_video)
                if key == ord("q"):
                    print("Stop at video:", i, "-", gaze_video)
                    break
                elif key == ord("s"):
                    with open(notable_videos_path, "a") as f:
                        f.write(f"{gaze_video}\n")

# gaze_video.split("\\")[-1].split(".mp4")[0]
