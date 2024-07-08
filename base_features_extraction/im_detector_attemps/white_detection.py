import cv2
import numpy as np
import base_features_extraction.im_detector_attemps.instruction_manual_detector as instruction_manual_detector
import os


def b(image):
    _, mask = instruction_manual_detector.get_white_area(image)
    gray_image = instruction_manual_detector.apply_mask(image, mask)

    # Applica la sogliatura per ottenere una maschera binaria
    _, binary_mask = cv2.threshold(gray_image, 235, 255, cv2.THRESH_BINARY)
    """_, binary_mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )"""

    # Trova i contorni nell'immagine binaria
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtra i contorni per dimensione (per escludere piccoli dettagli)
    min_contour_area = 1000
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
    ]

    # Disegna i contorni sulle immagini originali
    result_image = image.copy()
    cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)

    # Mostra l'immagine risultante
    return result_image


def c(img):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10
    )

    # Disegna le linee sull'immagine originale
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


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
        gaze_video, mask = instruction_manual_detector.compare_images(d_path)
        video = instruction_manual_detector.VideoReader(gaze_video)

        if video.video is None:
            continue

        frame = video.get_frame()

        cv2.imshow("Rilevamento tavolo", b(frame))
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord("q"):
            print("Stop at video:", i, "-", gaze_video)
            break
