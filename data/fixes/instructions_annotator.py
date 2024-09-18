import os
import cv2
import numpy as np
import json
from utils.classes import VideoReader


def get_white_area(image):
    _, threshold = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)

    return area


def get_table_videos_ranking(folder_path: str):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    areas_and_videos = {
        file: get_white_area(VideoReader(os.path.join(folder_path, file)).get_frame())
        for file in files
        if file.endswith(".mp4")
    }

    # Sort the videos by the white area (largest first)
    videos = sorted(areas_and_videos, key=areas_and_videos.get, reverse=True)

    return videos


def choose_video(folder_path: str):
    videos = get_table_videos_ranking(folder_path)

    chose_video = None
    count = 0
    video = VideoReader(os.path.join(folder_path, videos[count]))
    while not chose_video:
        ret, frame = video.read()
        if not ret:
            print("Next video")
            count = (count + 1) % len(videos)
            cv2.destroyAllWindows()
            continue
        cv2.imshow("Chose video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("e"):
            break
        elif key == ord("s"):
            chose_video = videos[count]
        elif key == ord("b") or key == ord("n"):
            if key == ord("b"):
                print("Previous video")
                count = max(count - 1, 0)
            else:
                count = (count + 1) % len(videos)
                cv2.destroyAllWindows()
            video = VideoReader(os.path.join(folder_path, videos[count]))
    cv2.destroyAllWindows()

    return chose_video, key


class BBoxDrawer:
    def __init__(self, image):
        self.image = image
        self.top_left_pt, self.bottom_right_pt = (-1, -1), (-1, -1)

    def draw_bbox(self, event, x: float, y: float, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.top_left_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.bottom_right_pt = (x, y)

    def get_bbox_coordinates(self):
        if self.top_left_pt != (-1, -1) and self.bottom_right_pt != (-1, -1):
            width = abs(self.bottom_right_pt[0] - self.top_left_pt[0])
            height = abs(self.bottom_right_pt[1] - self.top_left_pt[1])
            return self.top_left_pt[0], self.top_left_pt[1], width, height
        else:
            return None

    def run(self):
        cv2.namedWindow("Draw bbox")
        cv2.setMouseCallback("Draw bbox", self.draw_bbox)

        while True:
            cv2.imshow("Draw bbox", self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") or key == ord("q"):
                break
            elif key == ord("o"):
                self.top_left_pt, self.bottom_right_pt = (-1, -1), (-1, -1)

        cv2.destroyAllWindows()

        return key

    @staticmethod
    def get_bbox_from_video(video):
        frame = video.get_frame()
        bbox_drawer = BBoxDrawer(frame)
        key = bbox_drawer.run()
        return bbox_drawer.get_bbox_coordinates(), frame, key


def draw_bbox(image, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(image, p1, p2, (0, 255, 0), 2, 1)


class BBoxAnnotator:
    def __init__(self, videos_path: str, annotations_path: str):
        self.videos_path = videos_path
        self.annotations_path = annotations_path
        try:
            with open(annotations_path, "r") as f:
                self.annotations = json.load(f)
        except FileNotFoundError:
            self.annotations = {}
        self.old_annotations = json.loads(json.dumps(self.annotations))
        self.actual_video = None
        self.actual_tracker = None
        self.actual_bboxes_history = []
        self.actual_bbox = None

    def select_new_bbox(self, directory, camera_name, replace=False):
        key = 0
        tmp_bbox = [-1, -1, -1, -1]
        # While loop which keep going until key become "q" or key become "s" and tmp_bbox[0] != -1
        while key != ord("q") and not (key == ord("s") and tmp_bbox[0] != -1):
            tmp_bbox, frame, key = BBoxDrawer.get_bbox_from_video(self.actual_video)

        if key == ord("s"):
            self.actual_bbox = tmp_bbox

            to_save = (self.actual_video.get_index_frame(), self.actual_bbox)
            if replace:
                self.actual_bboxes_history.append(to_save)
            else:
                self.actual_bboxes_history[-1] = to_save

            self.actual_tracker = cv2.legacy.TrackerMIL_create()
            self.actual_tracker.init(frame, self.actual_bbox)

            self.annotations[directory][str(self.actual_video.get_index_frame())][
                camera_name
            ] = list(self.actual_bbox)

        return key

    def run(self):
        directories = os.listdir(self.videos_path)
        total_videos = len(directories)
        try:
            for i, d in enumerate(directories):

                if d in self.annotations:
                    user_input = input(
                        "Directory "
                        + d
                        + " already annotated, overwrite? (y/<empty>): "
                    )
                    if user_input.lower() != "y":
                        continue

                d_path = os.path.join(self.videos_path, d)
                if os.path.isdir(d_path):
                    print("Video directory:", str(i + 1), "/", total_videos, "-", d)
                    if d in self.annotations:
                        print(" - Restore last annotation")
                        camera_name = list(
                            self.annotations[d][list(self.annotations[d].keys())[0]]
                        )[0]
                        video_path = "HMC_" + camera_name.replace(":", "_") + ".mp4"
                    else:
                        print(" - Choose video to annotate")
                        video_path, key = choose_video(d_path)
                        if key == ord("q"):
                            print("Exiting...")
                            break
                        if video_path is None:
                            print("No video selected")
                            continue
                        camera_name = (
                            video_path.split("_")[1]
                            + ":"
                            + video_path.split("_")[2].split(".")[0]
                        )

                    self.actual_video = VideoReader(
                        os.path.join(d_path, video_path), grey_scale=False
                    )
                    if d in self.annotations:
                        self.actual_video.set_frame(
                            int(list(self.annotations[d].keys())[-1])
                        )
                        self.actual_bbox = self.annotations[d][
                            str(int(self.actual_video.get_index_frame()))
                        ][camera_name]
                    else:
                        self.actual_bbox, _, key = BBoxDrawer.get_bbox_from_video(
                            self.actual_video
                        )
                        if key == ord("q"):
                            print("Exiting...")
                            break
                        if self.actual_bbox is None:
                            print("Bounding box not selected for video:", video_path)
                            continue
                        self.annotations[d] = {
                            "0": {
                                camera_name: list(self.actual_bbox),
                            }
                        }

                    print(" - Annotating video")
                    # Creazione del tracker MIL
                    self.actual_tracker = cv2.legacy.TrackerMIL_create()
                    ret, frame = self.actual_video.read()
                    self.actual_tracker.init(frame, self.actual_bbox)

                    # Iterazione sui frame del video
                    self.actual_bboxes_history = [(0, self.actual_bbox)]
                    ret = True
                    wait = 0
                    while ret:
                        key = ""
                        ret, frame = self.actual_video.read()

                        # Tracking del bounding box
                        ok, self.actual_bbox = self.actual_tracker.update(frame)

                        # Disegno del bounding box
                        if ok:
                            draw_bbox(frame, self.actual_bbox)
                            self.actual_bboxes_history.append(
                                (self.actual_video.get_index_frame(), self.actual_bbox)
                            )
                        else:
                            print(" - Tracking failed, draw a new bounding box")
                            self.actual_video.previous_frame()
                            key = self.select_new_bbox(d, camera_name)

                        # Visualizzazione del frame
                        cv2.imshow("Check bbox", frame)
                        if key == "":
                            key = cv2.waitKey(wait)
                        if key == ord("q"):
                            print("Exiting...")
                            break
                        elif key == ord("p"):
                            wait = 0
                        elif key == ord("c"):
                            wait = 1
                        elif key == ord("b"):
                            self.actual_video.previous_frame()
                            self.actual_video.previous_frame()
                            self.actual_video.previous_frame()
                            if self.actual_bboxes_history:
                                self.actual_bboxes_history.pop()
                                _, self.actual_bbox = self.actual_bboxes_history.pop()
                                self.actual_tracker = cv2.legacy.TrackerMIL_create()
                                _, frame = self.actual_video.read()
                                self.actual_tracker.init(frame, self.actual_bbox)
                        elif key == ord("s"):
                            self.actual_video.previous_frame()
                            key = self.select_new_bbox(d, camera_name, True)
                            if key == ord("q"):
                                print("Exiting...")
                                break
                            wait = 1
                        elif key == ord("r"):
                            # Remove last bbox added
                            self.annotations[d].pop(
                                list(self.annotations[d].keys())[-1]
                            )

                    cv2.destroyAllWindows()
                    self.actual_video.release()

                if key == ord("q"):
                    break
        except Exception as e:
            print("Error:", e)

        if self.annotations == self.old_annotations:
            print("No new annotations")
        else:
            print("Saving annotations")
            with open(annotations_path, "w") as f:
                json.dump(self.annotations, f)

        if key != ord("q"):
            print("Done")


if __name__ == "__main__":
    directory_path = "D:/data/ego_recordings"
    annotations_path = "data/dump/instructions_annotations.json"
    BBoxAnnotator(directory_path, annotations_path).run()
