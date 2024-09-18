import atexit
import cv2
from enum import Enum
import json
from mmpose.apis import MMPoseInferencer
import numpy as np

from utils.classes import Source, FileSource, ChannelHandler
from utils.constants import log_manager
from utils.enums import DisplayLevel, LogCode, SensorMode, SourceMode
from utils.functions import load_dump, np_default


class PoseOpCode(Enum):
    """Enum class representing gaze Operation Codes in logs."""

    POSE_LOAD = 0
    NOT_FORCED_DATA = 1
    ESTIMATION = 2
    POSE_DUMP = 3


class PoseEstimator:
    def __init__(self, pose_source: Source = None, show=False):

        default = True
        if isinstance(pose_source, FileSource):
            if pose_source.mode == SourceMode.DUMP:
                try:
                    self.pose_estimation = ChannelHandler(
                        SensorMode.OFFLINE_DUMP,
                        load_dump(pose_source.path),
                        pose_source.path,
                        pose_source.new_dump,
                    )
                    first_frame = list(self.pose_estimation.data.keys())[0]
                    self.camera_name = list(
                        self.pose_estimation.data[first_frame].keys()
                    )[0]
                    self.pose_inferencer = None
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        PoseOpCode.POSE_LOAD.value,
                        "Dumped pose estimation loaded successfully",
                        DisplayLevel.LOW,
                    )
                    return
                except:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        PoseOpCode.POSE_LOAD.value,
                        "Pose estimation dump not loaded",
                        DisplayLevel.LOW,
                    )
            elif pose_source.mode == SourceMode.VIDEO_REF:
                try:
                    self.pose_inferencer = MMPoseInferencer("wholebody")
                    self.pose_estimation = ChannelHandler(
                        SensorMode.OFFLINE,
                        {},
                        pose_source.path,
                        pose_source.new_dump,
                    )
                    path_splitted = pose_source.path.split("/")
                    self.camera_name = (
                        path_splitted[-1].split("_")[-2]
                        + ":"
                        + path_splitted[-1].split("_")[-1].split(".")[0]
                    )
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        PoseOpCode.POSE_LOAD.value,
                        "Pose estimation video loaded successfully",
                        DisplayLevel.LOW,
                    )
                    default = False
                except:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        PoseOpCode.POSE_LOAD.value,
                        "Pose estimation video not loaded",
                        DisplayLevel.LOW,
                    )

        if default:
            self.pose_estimation = ChannelHandler()
            self.pose_inferencer = None
            self.camera_name = None
            self.show = False
            self.wait = None
        else:
            self.show = show
            self.wait = 1

        if self.is_initialized():
            atexit.register(self.dump_data)

    def is_initialized(self):
        return (
            self.pose_estimation.mode != None
            and self.camera_name != None
            and (
                self.pose_estimation.mode == SourceMode.DUMP
                or self.pose_inferencer != None
            )
        )

    def inference_pose(
        self,
        frame_index: float,
        frame=None,
        force=False,
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
        str_frame_index = str(frame_index)
        if self.pose_estimation.mode == SensorMode.OFFLINE_DUMP and not force:
            log_manager.log(
                self.__class__.__name__,
                LogCode.SUCCESS,
                PoseOpCode.NOT_FORCED_DATA,
                "Old data returned",
            )
            person = self.pose_estimation.data[str_frame_index][self.camera_name]
        elif self.pose_estimation.mode == SensorMode.OFFLINE and frame is not None:
            try:
                result_generator = self.pose_inferencer(frame, show=True, wait_time=1)
                result = next(result_generator)

                # result["predictions"][0].sort(key=lambda x: x["bbox_score"], reverse=True)
                person = result["predictions"][0][0]
                if person is not None:
                    self.pose_estimation.data[str_frame_index] = {
                        self.camera_name: person
                    }
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    PoseOpCode.ESTIMATION.value,
                    "Pose inferenced correctly",
                )
            except:
                pass

        if person is None:
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                PoseOpCode.ESTIMATION.value,
                "Pose inference failed",
                DisplayLevel.LOW,
            )

        return person

    def dump_data(self):
        outcome = False
        gaze_dump_path = self.pose_estimation.get_dump_path()
        if gaze_dump_path:
            try:
                f = open(gaze_dump_path, "w")
                json.dump(self.pose_estimation.data, f, default=np_default)
                f.close()
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    PoseOpCode.POSE_DUMP.value,
                    "Pose estimations dumped correctly",
                    DisplayLevel.LOW,
                )
                outcome = True
            except:
                pass

        if not outcome:
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                PoseOpCode.POSE_DUMP.value,
                "Pose estimations not dumped",
                DisplayLevel.LOW,
            )

        return outcome


if __name__ == "__main__":
    pose_source = FileSource(
        SourceMode.VIDEO_REF,
        "data/video_examples/nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10118_rgb.mp4",
        "data/dump/pose_dump.json",
    )
    pose_estimator = PoseEstimator(pose_source, show=True)

    if pose_estimator.is_initialized():
        cap = cv2.VideoCapture(pose_source.path)
        frame_index = 0
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                person = pose_estimator.inference_pose(frame_index, frame)

                keypoints = np.array([person["keypoints"]])
                keypoint_scores = np.array([person["keypoint_scores"]])

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

            frame_index += 1

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
