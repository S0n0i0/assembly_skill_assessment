import atexit
import cv2
from enum import Enum
import json
import numpy as np

from utils.classes import (
    BoundingBox,
    ChannelHandler,
    Source,
    PathSource,
    DataSource,
    VideoReader,
)
from utils.constants import log_manager
from utils.enums import DisplayLevel, LogCode, SensorMode, SourceMode
from utils.functions import get_derived_SM, load_dump, compute_distance


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
        gaze_source: Source | None = None,
        object_source: Source | None = None,
        gaze_target: Source | None = None,
    ):
        self.set_gaze(gaze_source, object_source, gaze_target)

        self.last_coordinates: dict[float, tuple[float, float]] = {}

        if self.is_initialized():
            atexit.register(self.dump_data)

    def set_gaze(
        self,
        gaze_source: Source | None = None,
        object_source: Source | None = None,
        gaze_target: Source | None = None,
    ):
        default = True
        if gaze_target.mode == SourceMode.DUMP:
            try:
                if isinstance(gaze_target, PathSource):
                    self.gaze_target = ChannelHandler(
                        SensorMode.OFFLINE_DUMP,
                        load_dump(gaze_target.path),
                        gaze_target.path,
                        gaze_source.new_dump,
                    )
                elif isinstance(gaze_target, DataSource):
                    self.gaze_target = ChannelHandler(
                        SensorMode.OFFLINE_DUMP,
                        gaze_target.data,
                        gaze_target.params.path,
                        gaze_source.new_dump,
                    )
                else:
                    raise Exception("Invalid gaze target source")
                self.gaze = ChannelHandler()
                self.objects = ChannelHandler()
                self.last_coordinates = None
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.GAZE_TARGET_LOAD.value,
                    "Dumped gaze target loaded successfully",
                    DisplayLevel.LOW,
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.GAZE_TARGET_LOAD.value,
                    "Gaze target not loaded",
                    DisplayLevel.LOW,
                )
        elif gaze_target.mode == SourceMode.DUMP and gaze_target.is_ref:
            self.gaze_target = ChannelHandler(
                None,
                {},
                gaze_target.path,
                gaze_target.new_dump,
            )
            log_manager.log(
                self.__class__.__name__,
                LogCode.SUCCESS,
                GazeOpCode.GAZE_TARGET_LOAD.value,
                "Gaze target initialized successfully",
                DisplayLevel.LOW,
            )
            default = False

        if default:
            self.gaze_target = ChannelHandler()

        default = True
        if gaze_source is not None and object_source is not None:
            tmp_frame = None
            if gaze_source.mode == SourceMode.VIDEO and gaze_source.is_ref:
                try:
                    if isinstance(gaze_source, PathSource):
                        self.camera_name, self.gaze = (
                            GazeObjectTracking.get_gaze_source_data(gaze_source)
                        )
                        if self.camera_name is None or self.gaze is None:
                            raise Exception("Video not loaded")
                    elif isinstance(gaze_source, DataSource):
                        self.camera_name, self.gaze = gaze_source.data
                    else:
                        raise Exception("Invalid gaze source")
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        GazeOpCode.GAZE_LOAD.value,
                        "Gaze video ref loaded successfully",
                        DisplayLevel.LOW,
                    )
                    default = False
                except:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        GazeOpCode.GAZE_LOAD.value,
                        "Gaze video ref not loaded",
                        DisplayLevel.LOW,
                    )
            elif gaze_source.mode == SourceMode.DUMP:
                try:
                    if isinstance(gaze_source, PathSource):
                        self.gaze = ChannelHandler(
                            SensorMode.OFFLINE_DUMP,
                            load_dump(gaze_source.path),
                            gaze_source.path,
                            gaze_source.new_dump,
                        )
                    elif isinstance(gaze_source, DataSource):
                        self.gaze = ChannelHandler(
                            SensorMode.OFFLINE_DUMP,
                            gaze_source.data,
                            gaze_source.params.path,
                            gaze_source.new_dump,
                        )
                    else:
                        raise Exception("Invalid gaze source")
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        GazeOpCode.GAZE_LOAD.value,
                        "Gaze dump loaded successfully",
                        DisplayLevel.LOW,
                    )
                    default = False
                except:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        GazeOpCode.GAZE_LOAD.value,
                        "Gaze dump not loaded",
                        DisplayLevel.LOW,
                    )

            if not default:

                if (
                    object_source.mode == SourceMode.DUMP
                    or object_source.mode == SourceMode.SUPPORT_DUMP
                ):
                    try:
                        mode = (
                            SensorMode.OFFLINE_DUMP
                            if object_source.mode == SourceMode.DUMP
                            else SensorMode.OFFLINE
                        )
                        if isinstance(object_source, PathSource):
                            self.objects = ChannelHandler(
                                mode,
                                load_dump(object_source.path)[
                                    self.gaze.data["sequence_name"]
                                ],
                                object_source.path,
                                object_source.new_dump,
                            )
                        elif isinstance(object_source, DataSource):
                            self.objects = ChannelHandler(
                                mode,
                                object_source.data[self.gaze.data["sequence_name"]],
                                object_source.params.path,
                                object_source.new_dump,
                            )
                        if tmp_frame is None and gaze_source.mode != SourceMode.DUMP:
                            raise Exception("Gaze data not loaded")
                        first_frame = list(self.objects.data.keys())[0]
                        tmp_camera = list(self.objects.data[first_frame].keys())[0]
                        if self.camera_name == tmp_camera:
                            self.tracker = cv2.legacy.TrackerMIL_create()
                            self.tracker.init(
                                tmp_frame,
                                self.objects.data[first_frame][self.camera_name],
                            )
                            log_manager.log(
                                self.__class__.__name__,
                                LogCode.SUCCESS,
                                GazeOpCode.OBJECTS_LOAD.value,
                                "Dumped objects data loaded successfully",
                                DisplayLevel.LOW,
                            )
                            default = False
                    except:
                        log_manager.log(
                            self.__class__.__name__,
                            LogCode.ERROR,
                            GazeOpCode.OBJECTS_LOAD.value,
                            "Objects data dump not loaded",
                            DisplayLevel.LOW,
                        )
                        default = True

        if default:
            self.gaze = ChannelHandler()
            self.objects = ChannelHandler()
            self.camera_name = None
            self.tracker = None

        tmp_mode = get_derived_SM(self.gaze.mode, self.objects.mode)
        self.gaze_target.mode = (
            tmp_mode if tmp_mode is not SensorMode.OFFLINE_DUMP else SensorMode.OFFLINE
        )

    def get_gaze_source_data(gaze_source: PathSource):
        tmp_video = VideoReader(gaze_source.path)
        ret, tmp_frame = tmp_video.read()
        if not ret:
            return None, None
        path_splitted = gaze_source.path.split("/")
        camera_name = (
            path_splitted[-1].split("_")[-2]
            + ":"
            + path_splitted[-1].split("_")[-1].split(".")[0]
        )
        gaze = ChannelHandler(
            SensorMode.OFFLINE,
            {
                "sequence_name": path_splitted[-2],
                "camera_name": camera_name,
                "width": tmp_frame.shape[1],
                "height": tmp_frame.shape[0],
                "gaze_coordinates": (
                    tmp_frame.shape[1] // 2,
                    tmp_frame.shape[0] // 2,
                ),
                "proximity_threshold": gaze_source.params["proximity_threshold"],
                "approaching_threshold": gaze_source.params["approaching_threshold"],
                "coordinates_memory_size": gaze_source.params[
                    "coordinates_memory_size"
                ],
                "correct_directions": gaze_source.params["correct_directions"],
            },
            gaze_source.path,
            gaze_source.new_dump,
        )
        return camera_name, gaze

    def is_initialized(self):
        return (
            self.gaze_target.mode != None
            and self.gaze.mode != None
            and self.objects.mode != None
            and self.camera_name != None
            and (self.objects.mode == SourceMode.DUMP or self.tracker != None)
        )

    def memorize_coordinates(
        self, frame_index: float, coordinates: tuple[float, float]
    ):
        if self.gaze.data["coordinates_memory_size"] == 0:
            return
        if (
            len(self.last_coordinates.keys())
            == self.gaze.data["coordinates_memory_size"]
        ):
            self.last_coordinates.pop(min(self.last_coordinates.keys()))
        self.last_coordinates[frame_index] = coordinates

    def is_object_coming(self) -> bool:
        if len(self.last_coordinates) == 0:
            return None

        correct_directions = 0
        frames = [
            self.last_coordinates[key] for key in sorted(self.last_coordinates.keys())
        ]
        previous_distance = compute_distance(
            self.gaze.data["gaze_coordinates"], frames[0]
        )
        for f in frames[1:]:
            actual_distance = compute_distance(self.gaze.data["gaze_coordinates"], f)
            if actual_distance < previous_distance:
                correct_directions += 1
            previous_distance = actual_distance

        return correct_directions >= self.gaze.data["correct_directions"]

    def compute_gaze_target(
        self,
        frame_index: float,
        frame=None,
        force=False,
    ):
        if not self.is_initialized():
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                GazeOpCode.TRACKING.value,
                "GazeObjectTracking not initialized correctly",
                DisplayLevel.LOW,
            )
            return None

        str_frame_index = str(frame_index)
        if self.gaze_target.mode == SensorMode.OFFLINE_DUMP and not force:
            log_manager.log(
                self.__class__.__name__,
                LogCode.SUCCESS,
                GazeOpCode.NOT_FORCED_DATA,
                "Old data returned",
            )
            return self.gaze_target.data[str_frame_index][self.camera_name]

        object_bbox = None
        if self.objects.mode == SensorMode.OFFLINE_DUMP:
            object_bbox = BoundingBox(
                self.objects.data[str_frame_index][self.camera_name]
            )
        elif self.objects.mode == SensorMode.OFFLINE and frame is not None:
            if (
                str_frame_index in self.objects.data
                and self.camera_name in self.objects.data[str_frame_index]
            ):
                object_bbox = BoundingBox(
                    self.objects.data[str_frame_index][self.camera_name]
                )
                self.tracker = cv2.legacy.TrackerMIL_create()
                self.tracker.init(frame, object_bbox.to_xywh())
            else:
                ok, object_bbox = self.tracker.update(frame)
                if not ok:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        GazeOpCode.TRACKING.value,
                        "Objects tracking failed",
                        DisplayLevel.LOW,
                    )
                    return None
                self.objects.data[str_frame_index] = {self.camera_name: object_bbox}
                object_bbox = BoundingBox(object_bbox)

        object_target = None
        if object_bbox is not None:
            self.memorize_coordinates(frame_index, object_bbox.get_center())
            if (
                object_bbox.get_distance(self.gaze.data["gaze_coordinates"])
                < self.gaze.data["approaching_threshold"]
                and self.is_object_coming()
            ) or object_bbox.get_distance(
                self.gaze.data["gaze_coordinates"]
            ) < self.gaze.data[
                "proximity_threshold"
            ]:
                object_target = GazeTarget.INSTRUCTION_MANUAL.value
            else:
                object_target = GazeTarget.OTHER.value

            self.gaze_target.data[str_frame_index] = {self.camera_name: object_target}
            log_manager.log(
                self.__class__.__name__,
                LogCode.SUCCESS,
                GazeOpCode.TRACKING.value,
                "Gaze target tracked correctly",
            )

        return object_target

    def dump_data(self) -> tuple[bool, bool, bool]:
        outcome = [False, False, False]
        gaze_dump_path = self.gaze.get_dump_path()
        if gaze_dump_path:
            try:
                f = open(gaze_dump_path, "w")
                json.dump(self.gaze.data, f)
                f.close()
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.GAZE_DUMP.value,
                    "Gaze dumped correctly",
                    DisplayLevel.LOW,
                )
                outcome[0] = True
            except:
                pass

        if not outcome[0]:
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                GazeOpCode.GAZE_DUMP.value,
                "Gaze not dumped",
                DisplayLevel.LOW,
            )

        objects_dump_path = self.objects.get_dump_path()
        if objects_dump_path:
            try:
                with open(objects_dump_path, "w") as f:
                    json.dump(self.objects.data, f)
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.OBJECTS_DUMP.value,
                    "Objects dumped correctly",
                    DisplayLevel.LOW,
                )
                outcome[1] = True
            except:
                pass

        if not outcome[1]:
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                GazeOpCode.OBJECTS_DUMP.value,
                "Objects not dumped",
                DisplayLevel.LOW,
            )

        gaze_target_dump_path = self.gaze_target.get_dump_path()
        if gaze_target_dump_path:
            try:
                with open(gaze_target_dump_path, "w") as f:
                    json.dump(self.gaze_target.data, f)
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.GAZE_TARGET_DUMP.value,
                    "Gaze target dumped correctly",
                    DisplayLevel.LOW,
                )
                outcome[2] = True
            except:
                pass

        if not outcome[2]:
            log_manager.log(
                self.__class__.__name__,
                LogCode.ERROR,
                GazeOpCode.GAZE_TARGET_DUMP.value,
                "Gaze target not dumped",
                DisplayLevel.LOW,
            )

        return tuple(outcome)


GazeObjectTracking.get_gaze_source_data = staticmethod(
    GazeObjectTracking.get_gaze_source_data
)

if __name__ == "__main__":
    mode = 1
    if mode == 0:
        ed = PathSource(
            SourceMode.DUMP,
            False,
            "../nusar-2021_action_both_9065-b05a_9095_user_id_2021-02-17_122813_e.json",
        )
        pd = PathSource(
            SourceMode.DUMP,
            False,
            "../nusar-2021_action_both_9065-b05a_9095_user_id_2021-02-17_122813_p.json",
        )
        a = GlobalGazeTracking(ed, pd)
        cameras_dump = PathSource(SourceMode.DUMP, False, "../cameras_dump.json", True)
        gaze_dump = PathSource(SourceMode.DUMP, False, "../gaze_dump.json", True)

        a.compute_world_data(False, cameras_dump, gaze_dump, 2)
    elif mode == 1:
        gaze = PathSource(
            SourceMode.VIDEO,
            True,
            "data/video_examples/nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/HMC_84358933_mono10bit.mp4",
            "data/dump/gaze_dump.json",
            {
                "proximity_threshold": 40,
                "approaching_threshold": 60,
                "coordinates_memory_size": 15,
                "correct_directions": 10,
            },
        )
        objects = PathSource(
            SourceMode.SUPPORT_DUMP,
            False,
            "data/dump/instructions_annotations.json",
            "data/dump/objects_dump.json",
        )
        gaze_target = PathSource(
            SourceMode.DUMP, True, "data/dump/object_target_dump.json"
        )
        gaze_target_tracker = GazeObjectTracking(
            gaze,
            objects,
            gaze_target,
        )

        if gaze_target_tracker.is_initialized():
            cap = VideoReader(gaze.path, 0, False)
            ret = True
            frame_index = 0
            while ret:
                ret, frame = cap.read()

                target = gaze_target_tracker.compute_gaze_target(frame_index, frame)

                # write gaze_target on the frame and show it
                cv2.putText(
                    frame,
                    f"Gaze Target: {target}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Frame", frame)

                frame_index += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
