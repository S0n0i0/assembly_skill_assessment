import cv2
from enum import Enum
import json
import numpy as np
import os
import pickle
import tqdm

from utils.classes import (
    BoundingBox,
    Source,
    PathSource,
)
from utils.constants import log_manager
from utils.enums import DisplayLevel, LogCode, SensorMode, SourceMode
from utils.functions import load_dump, add_to_dump, dump_data


class GazeOpCode(Enum):
    """Enum class representing gaze Operation Codes in logs."""

    GAZE_LOAD = 0
    CAMERAS_WORLD_LOAD = 1
    CAMERAS_LOAD = 2
    GAZE_TARGET_LOAD = 3
    OBJECTS_LOAD = 4
    NOT_FORCED_DATA = 5
    FORCE_DATA = 6
    TRACKING = 7
    GAZE_DUMP = 8
    OBJECTS_DUMP = 9
    GAZE_TARGET_DUMP = 10


class GlobalGazeSubmode(Enum):
    """
    Enum class representing the submodes of gaze tracking.

    Attributes:
        GAZE_DUMP (str): Dump gaze data.
        WORLD_CAMERAS_DUMP (str): Dump world coordinates for cameras data.
        INPUT_DUMP (str): Dump extrinsic and position data of cameras data.
    """

    GAZE_DUMP = "GAZE_DUMP"
    WORLD_CAMERAS_DUMP = "WORLD_CAMERAS_DUMP"
    INPUT_DUMP = "INPUT_DUMP"


class GazeTarget(Enum):
    INSTRUCTION_MANUAL = "instruction_manual"
    OTHER = "other"


class GlobalGazeTracking:
    """Class responsible for gaze tracking analysis.

    This class is used to track the gaze of a user by analyzing the positions of multiple cameras.
    It provides methods to calculate the gaze vector of the user and compute the world coordinates of the cameras.

    Attributes:
        extrinsic_data (dict): Extrinsic parameters of the cameras.
        position_data (dict): Positions of the cameras.
        cameras_world_data (dict): World coordinates of the cameras.
    """

    def __init__(
        self,
        extrinsic_source: Source | None = None,
        position_source: Source | None = None,
        cameras_world_dump_path: str | None = None,
        gaze_world_dump_puth: str | None = None,
    ):
        """Initialize the GazeTracking object.

        Args:
            extrinsic_source (utils.classes.Source): Object containing information about the extrinsic parameters source.
            position_source (utils.classes.Source): Object containing information about the position parameters source.
            cameras_world_dump_path (str): Path of the coordinates of cameras w.r.t. the world.
            gaze_world_dump_puth (str): Path of coordinates of user's gaze w.r.t. the world
        """

        default = True
        if gaze_world_dump_puth:
            try:
                self.gaze_world_data = load_dump(gaze_world_dump_puth)
                self.gaze_world_path = gaze_world_dump_puth
                self.cameras_world_data = None
                self.extrinsic_data = None
                self.position_data = None
                self.mode = SensorMode.OFFLINE_DUMP
                self.submode = GlobalGazeSubmode.GAZE_DUMP
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.GAZE_LOAD.value,
                    "Dumped gaze world coordinates loaded successfully",
                    DisplayLevel.LOW,
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.GAZE_LOAD.value,
                    "Gaze world coordinates not loaded",
                    DisplayLevel.LOW,
                )
        if default:
            self.gaze_world_path = None
            self.gaze_world_data = None

        default = True
        if cameras_world_dump_path:
            try:
                self.cameras_world_data = load_dump(cameras_world_dump_path)
                self.world_path = cameras_world_dump_path
                self.extrinsic_data = None
                self.position_data = None
                self.mode = SensorMode.OFFLINE_DUMP
                self.submode = GlobalGazeSubmode.WORLD_CAMERAS_DUMP
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.CAMERAS_WORLD_LOAD.value,
                    "Dumped cameras world coordinates loaded successfully",
                    DisplayLevel.LOW,
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.CAMERAS_WORLD_LOAD.value,
                    "Cameras world coordinates not loaded",
                    DisplayLevel.LOW,
                )

        if default:
            self.world_path = None
            self.cameras_world_data = None

        default = True
        if (
            extrinsic_source
            and position_source
            and extrinsic_source.mode == SourceMode.DUMP
            and position_source.mode == SourceMode.DUMP
        ):
            try:
                self.extrinsic_data = load_dump(extrinsic_source.path)
                self.position_data = load_dump(position_source.path)
                self.extrinsic_path = extrinsic_source.path
                self.position_path = position_source.path
                self.mode = SensorMode.OFFLINE_DUMP
                self.submode = GlobalGazeSubmode.INPUT_DUMP
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.CAMERAS_LOAD.value,
                    "Dumped cameras data loaded successfully",
                    DisplayLevel.LOW,
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.CAMERAS_LOAD.value,
                    "Cameras data dump not loaded",
                    DisplayLevel.LOW,
                )

        if default:
            self.extrinsic_path = None
            self.position_path = None
            self.extrinsic_data = {}
            self.position_data = {}

    def compute_gaze_vector(self, c1: np.array, c2: np.array, c3: np.array, norm=False):
        """Calculate the gaze vector of the user.

        Args:
            c1 (np.array): Camera 1 world coordinates.
            c2 (np.array): Camera 2 world coordinates.
            c3 (np.array): Camera 3 world coordinates.
            norm (bool): If True, normalize the gaze vector.

        Returns:
            np.array: Gaze vector of the user.
        """

        # Calculate the gaze vector using the camera world coordinates
        gaze_vector = np.cross(c2 - c1, c3 - c1)

        return gaze_vector / np.linalg.norm(gaze_vector) if norm else gaze_vector

    def compute_world_data(
        self,
        norm=False,
        cameras_dump: PathSource | None = None,
        gaze_dump: PathSource | None = None,
        force=0,
    ):
        """Calculate the world coordinates of the cameras and derive the average point for gaze.

        Args:
            norm (bool): If True, normalize the gaze vectors.
            cameras_dump (utils.classes.FileSource): If specified, save the world coordinates in that file.
            gaze_dump (utils.classes.FileSource): If specified, save the gaze world vectors in that file.
            force (int): If 0 return old data. If 1, recalculate gaze vectors. If 2, recalculate both cameras world and gaze coordinates.

        Returns:
            tuple[dict[str, dict[str, list[list[float]]]], dict[str, list[list[float]]]]: A tuple containing the calculated cameras world coordinates and gaze vectors.
        """

        # If force is 0 return the cameras world and gaze coordinates which are already calculated
        if self.gaze_world_data and self.cameras_world_data and force == 0:
            log_manager.log(
                self.__class__.__name__,
                LogCode.SUCCESS,
                GazeOpCode.NOT_FORCED_DATA,
                "Old data returned",
            )
            return (self.cameras_world_data, self.gaze_world_data)
        else:
            self.gaze_world_data = {}

        # If force is 2 or the cameras world coordinates are not calculated, recalculate the cameras world coordinates
        if force == 2 or self.cameras_world_data is None:
            compute_cameras = True
            self.cameras_world_data = {}
            log_manager.log(
                self.__class__.__name__,
                LogCode.SUCCESS,
                GazeOpCode.FORCE_DATA,
                "Recalculating cameras world coordinates",
            )
        else:
            compute_cameras = False

        # Iterate over the frames and cameras to calculate the world coordinates
        for frame_id, cameras_positions in self.position_data.items():
            # If the cameras world coordinates are already calculated, skip the frame
            if compute_cameras:
                self.cameras_world_data[frame_id] = {}
                for camera_id, position in cameras_positions.items():
                    # Convert the position to a 4D vector with the fourth component set to 1
                    position_4d = np.append(position, 1)

                    # Get the corresponding extrinsic transformation matrix
                    extrinsic_matrix = np.array(
                        self.extrinsic_data[frame_id][camera_id]
                    )

                    # Compute the derived coordinates by multiplying the matrix by the position vector and convert back to 3d coordinates
                    self.cameras_world_data[frame_id][camera_id] = np.dot(
                        extrinsic_matrix, position_4d
                    )[0:3].tolist()

            # Compute gaze vector from the cameras world coordinates.
            cameras_ids = list(cameras_positions.keys())
            self.gaze_world_data[frame_id] = self.compute_gaze_vector(
                np.array(self.cameras_world_data[frame_id][cameras_ids[0]]),
                np.array(self.cameras_world_data[frame_id][cameras_ids[1]]),
                np.array(self.cameras_world_data[frame_id][cameras_ids[2]]),
                norm,
            ).tolist()
            break

        # Dump the world coordinates if requested
        if cameras_dump is not None and cameras_dump.mode == SourceMode.DUMP:
            try:
                with open(
                    cameras_dump.path, "w" if cameras_dump.new_dump else "a"
                ) as f:
                    json.dump(self.cameras_world_data, f)
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    3,
                    "Output cameras world coordinates dumped successfully",
                )
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    3,
                    "Dump of cameras world coordinates failed",
                    DisplayLevel.LOW,
                )

        # Dump the gaze vectors if requested
        if gaze_dump is not None and gaze_dump.mode == SourceMode.DUMP:
            try:
                with open(gaze_dump.path, "w" if gaze_dump.new_dump else "a") as f:
                    json.dump(self.gaze_world_data, f)
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    3,
                    "Output gaze world vectors dumped successfully",
                )
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    3,
                    "Dump of gaze world vectors failed",
                    DisplayLevel.LOW,
                )

        return (self.cameras_world_data, self.gaze_world_data)


class GazeObjectTracking:

    def __init__(
        self,
    ):
        self.gaze_source_memory: dict[str, dict[str, dict[str, any]]] = {}
        self.tracker = self.init_tracker()

        log_manager.log(
            self.__class__.__name__,
            LogCode.SUCCESS,
            GazeOpCode.TRACKING.value,
            "Objects tracking initialized correctly",
            DisplayLevel.LOW,
        )

    def init_tracker(
        frame: cv2.typing.MatLike | None = None, bbox: cv2.typing.Rect2d | None = None
    ):
        tracker: cv2.legacy.Tracker = cv2.legacy.TrackerMIL_create()
        if bbox is not None:
            tracker.init(
                frame,
                bbox,
            )

        return tracker

    def get_gaze_source_data(
        image: cv2.typing.MatLike,
        params: dict[str, float],
    ):
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "gaze_coordinates": (
                image.shape[1] // 2,
                image.shape[0] // 2,
            ),
            "proximity_threshold": params["proximity_threshold"],
            "approaching_threshold": params["approaching_threshold"],
            "correct_directions": params["correct_directions"],
        }

    def is_initialized(self):
        return self.tracker is not None and self.gaze_source_memory is not None

    def is_object_coming(
        self, gaze: dict[str, float], object_bboxes: list[BoundingBox]
    ) -> bool:
        if len(self.last_coordinates) == 0:
            return None

        correct_directions = 0
        previous_distance = object_bboxes[0].get_distance(gaze["gaze_coordinates"])
        for f in object_bboxes[1:]:
            actual_distance = f.get_distance(gaze["gaze_coordinates"])
            if actual_distance < previous_distance:
                correct_directions += 1
            previous_distance = actual_distance

        return correct_directions >= gaze["correct_directions"]

    def compute_gaze_target(
        self,
        sequence: str,
        view: str,
        offset: int,
        frames: list[cv2.typing.MatLike],
        params: dict[str, float],
        support_bboxes: dict[int, BoundingBox],
    ) -> tuple[dict[str, any], list[BoundingBox], str]:
        if not self.is_initialized():
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                GazeOpCode.TRACKING.value,
                "GazeObjectTracking not initialized correctly",
                DisplayLevel.LOW,
            )
            return {}, [], GazeTarget.OTHER.value

        if (
            sequence not in self.gaze_source_memory
            or view not in self.gaze_source_memory[sequence]
        ):
            gaze: dict[str, any] = self.get_gaze_source_data(frames[0], params)
            if sequence not in self.gaze_source_memory:
                self.gaze_source_memory[sequence] = {}
            self.gaze_source_memory[sequence][view] = gaze
        else:
            gaze: dict[str, any] = self.gaze_source_memory[sequence][view]

        object_bboxes: list[BoundingBox | None] = [None] * len(frames)
        for i, frame in enumerate(frames):
            if i in support_bboxes:
                object_bboxes[i] = support_bboxes[offset + i]
                self.tracker = self.init_tracker(frame, object_bboxes[i].to_xywh())

            ok, object_bbox = self.tracker.update(frame)
            if not ok:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.TRACKING.value,
                    "Objects tracking failed",
                    DisplayLevel.LOW,
                )
                continue
            object_bboxes[i] = BoundingBox(object_bbox)

        targets = [
            (
                GazeTarget.INSTRUCTION_MANUAL.value
                if object_bbox.get_distance(gaze["gaze_coordinates"])
                < gaze["proximity_threshold"]
                else GazeTarget.OTHER.value
            )
            for object_bbox in object_bboxes
        ]
        object_target = max(set(targets), key=targets.count)
        # If the target is OTHER but the object is coming, change the target to INSTRUCTION_MANUAL
        if object_target == GazeTarget.OTHER.value and (
            object_bboxes[-1].get_distance(gaze["gaze_coordinates"])
            < gaze["approaching_threshold"]
            and self.is_object_coming(gaze, object_bboxes)
        ):
            object_target = GazeTarget.INSTRUCTION_MANUAL.value

        log_manager.log(
            self.__class__.__name__,
            LogCode.SUCCESS,
            GazeOpCode.TRACKING.value,
            "Gaze target tracked correctly",
        )

        return gaze, object_bboxes, object_target


# GazeObjectTracking.get_gaze_source_data = staticmethod(
#     GazeObjectTracking.get_gaze_source_data
# )


def analize_video(path: str, support_bboxes: dict[int], params: dict[str, float]):
    global gaze_target_tracker

    cap = cv2.VideoCapture(path)
    object_source_data: list[BoundingBox] = [None] * int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gaze_target_data = [None] * int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // params["delta"])
    frame_index = 0
    target_index = 0
    # Loop through the video frames
    frames = []
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            frames.append(frame)
            if len(frames) > params["window_size"]:
                frames.pop(0)
            if target_index % params["delta"] != 0:
                gaze_source_data, object_bboxes, object_target = (
                    gaze_target_tracker.compute_gaze_target(
                        path.split("/")[-2],
                        path.split("/")[-1].replace(".mp4", ""),
                        frame_index,
                        frames,
                        params,
                        support_bboxes,
                    )
                )

                # Place elements of object_bboxes in the correct position in object_source
                for i, bbox in enumerate(object_bboxes):
                    object_source_data[frame_index + i] = bbox

                gaze_target_data[target_index] = object_target

                target_index += 1

            frame_index += 1
        else:
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    return gaze_source_data, object_source_data, gaze_target_data

if __name__ == "__main__":
    sources = {
        "gaze_source": PathSource(
        SourceMode.VIDEO,
        False,
        "D:/data/videos/fixed_recordings",
        "D:/data/dumps/gaze_analysis/gaze_source_dump.pkl",
        {
            "proximity_threshold": 40,
            "approaching_threshold": 60,
            "correct_directions": 10,
            "window_size": 15,
            "delta": 5,
        },
    ),
        "object_source": PathSource(
        SourceMode.SUPPORT_DUMP,
        False,
        "D:/data/dumps/gaze_analysis/instructions_annotations.pkl",
        "D:/data/dumps/gaze_analysis/object_source_dump.pkl",
    ),
        "gaze_target": PathSource(
        SourceMode.DUMP,
        False,
        "D:/data/dumps/gaze_analysis/gaze_target_dump.pkl",
    ),
    }
    gaze_target_tracker = GazeObjectTracking()
    force = {
        "gaze_source": False,
        "object_source": False,
        "gaze_target": False,
    }

    with open(sources["object_source"].path, "rb") as f:
        support_bboxes = pickle.load(f)

    dump = {
        "gaze_source": {},
        "object_source": {},
        "gaze_target": {},
    }
    if gaze_target_tracker.is_initialized():
        if os.path.isfile(sources["gaze_source"].path):
            print("Analyzing a video")
            dump["gaze_source"], dump["object_source"], dump["gaze_target"] = analize_video(
                sources["gaze_source"].path, support_bboxes, sources["gaze_source"].params
            )
            
            divided_path = sources["gaze_source"].get_dump_path().split("/")
            keys = (divided_path[-2], divided_path[-1].replace(".mp4", ""))
            for data in dump:
                add_to_dump(sources["gaze_source"], data, keys)
        else:
            print("Analyzing a directory")
            try:
                for sequence in tqdm(os.listdir(sources["gaze_source"].path)):
                    for data in dump:
                        if sequence not in data:
                            data[sequence] = {}

                    for view in os.listdir(f"{sources["gaze_source"].path}/{sequence}"):
                        view = view.replace(".mp4", "")
                        if (
                            not any(force.values())
                            and view in dump[sequence]
                            and dump[sequence][view][-1] != None
                        ):
                            continue
                        tmp_data = {
                            "gaze_source": {},
                            "object_source": {},
                            "gaze_target": {},
                        }
                        tmp_data["gaze_source"], tmp_data["object_source"], tmp_data["gaze_target"] = analize_video(
                            f"{sources["gaze_source"].path}/{sequence}/{view}.mp4", support_bboxes, sources["gaze_source"].params
                        )
                        for key in tmp_data:
                            dump[key][sequence][view] = tmp_data[key]
            except KeyboardInterrupt:
                pass

            for key in dump:
                dump_data(sources[key].get_dump_path(), dump[key], force[key])