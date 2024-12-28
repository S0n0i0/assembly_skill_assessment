import cv2
from enum import Enum
from mmpose.apis import MMPoseInferencer
import numpy as np
import os
from tqdm import tqdm

from utils.classes import PathSource
from utils.constants import log_manager
from utils.enums import DisplayLevel, LogCode, SourceMode
from utils.functions import add_to_dump, dump_data


class PoseOpCode(Enum):
    """Enum class representing gaze Operation Codes in logs."""

    POSE_LOAD = 0
    NOT_FORCED_DATA = 1
    ESTIMATION = 2
    POSE_DUMP = 3


class PoseEstimator:
    def __init__(self, show=False, wait_time=0):
        self.show = show
        self.wait_time = wait_time

        self.pose_inferencer = MMPoseInferencer("wholebody")
        log_manager.log(
            self.__class__.__name__,
            LogCode.SUCCESS,
            PoseOpCode.POSE_LOAD.value,
            "Pose estimator loaded successfully",
            DisplayLevel.LOW,
        )

    def is_initialized(self):
        return self.pose_inferencer != None

    def inference_pose(
        self,
        frame: cv2.typing.MatLike,
    ):
        if not self.is_initialized():
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                PoseOpCode.ESTIMATION.value,
                "PoseEstimator not initialized correctly",
                DisplayLevel.LOW,
            )
            return None

        person = None
        try:
            result_generator = self.pose_inferencer(
                frame, show=self.show, wait_time=self.wait_time
            )
            result = next(result_generator)

            # result["predictions"][0].sort(key=lambda x: x["bbox_score"], reverse=True)
            person = result["predictions"][0][0]
            log_manager.log(
                self.__class__.__name__,
                LogCode.SUCCESS,
                PoseOpCode.ESTIMATION.value,
                "Pose inferenced correctly",
            )
        except KeyboardInterrupt:
            return False
        except:
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                PoseOpCode.ESTIMATION.value,
                "Pose inference failed",
                DisplayLevel.LOW,
            )

        return person


def analize_video(path: str):
    global pose_estimator

    cap = cv2.VideoCapture(path)
    analysis = [None] * int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    # Loop through the video frames
    quit = False
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            person = pose_estimator.inference_pose(frame)

            if person is False:
                quit = True
                break
            if person is not None:
                analysis[frame_index] = {
                    "keypoints": np.array([person["keypoints"]]),
                    "keypoint_scores": np.array([person["keypoint_scores"]]),
                }

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit = True
                break
        else:
            break

        frame_index += 1

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    return analysis, quit


if __name__ == "__main__":
    pose_source = PathSource(
        SourceMode.VIDEO,
        True,
        "D:/data/videos/fixed_recordings",
        "D:/data/dumps/pose_estimation/pose_dump.pkl",
    )
    force = False
    pose_estimator = PoseEstimator()

    if pose_estimator.is_initialized():
        if os.path.isfile(pose_source.path):
            print("Analyzing a video")
            analysis, _ = analize_video(pose_source.path)
            add_to_dump(pose_source, analysis)
        else:
            print("Analyzing a directory")
            dump = {}
            try:
                for sequence in tqdm(os.listdir(pose_source.path)):
                    if sequence not in dump:
                        dump[sequence] = {}
                    for view in os.listdir(f"{pose_source.path}/{sequence}"):
                        view = view.replace(".mp4", "")
                        if (
                            not force
                            and view in dump[sequence]
                            and dump[sequence][view][-1] != None
                        ):
                            continue
                        dump[sequence][view], quit = analize_video(
                            f"{pose_source.path}/{sequence}/{view}.mp4"
                        )
                        if quit:
                            break
                    if quit:
                        break
            except KeyboardInterrupt:
                pass

            dump_data(pose_source, dump, force)
