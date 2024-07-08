from enum import Enum
import json
import numpy as np
from utils.classes import Source, FileSource
from utils.constants import log_manager
from utils.enums import LogCode, SourceMode, SensorMode
from utils.functions import load_dump


class GazeOpCode(Enum):
    """Enum class representing gaze Operation Codes in logs."""

    GAZE_DUMP = 0
    CAMERAS_WORLD_DUMP = 1
    CAMERAS_DUMP = 2
    NOT_FORCED_DATA = 3
    FORCE_DATA = 4


class GazeSubmode(Enum):
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


class GazeTracking:
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
        extrinsic_source: Source = None,
        position_source: Source = None,
        cameras_world_dump_path: str = None,
        gaze_world_dump_puth: str = None,
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
                self.submode = GazeSubmode.GAZE_DUMP
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.GAZE_DUMP.value,
                    "Dumped gaze world coordinates loaded successfully",
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.GAZE_DUMP.value,
                    "Gaze world coordinates not loaded",
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
                self.submode = GazeSubmode.WORLD_CAMERAS_DUMP
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.CAMERAS_WORLD_DUMP.value,
                    "Dumped cameras world coordinates loaded successfully",
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.CAMERAS_WORLD_DUMP.value,
                    "Cameras world coordinates not loaded",
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
                self.submode = GazeSubmode.INPUT_DUMP
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    GazeOpCode.CAMERAS_DUMP.value,
                    "Dumped cameras data loaded successfully",
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    GazeOpCode.CAMERAS_DUMP.value,
                    "Cameras data dump not loaded",
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
        cameras_dump: FileSource = None,
        gaze_dump: FileSource = None,
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
                    cameras_dump.path, "w" if cameras_dump.new_file else "a"
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
                )

        # Dump the gaze vectors if requested
        if gaze_dump is not None and gaze_dump.mode == SourceMode.DUMP:
            try:
                with open(gaze_dump.path, "w" if gaze_dump.new_file else "a") as f:
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
                )

        return (self.cameras_world_data, self.gaze_world_data)


if __name__ == "__main__":
    ed = FileSource(
        SourceMode.DUMP,
        "../nusar-2021_action_both_9065-b05a_9095_user_id_2021-02-17_122813_e.json",
    )
    pd = FileSource(
        SourceMode.DUMP,
        "../nusar-2021_action_both_9065-b05a_9095_user_id_2021-02-17_122813_p.json",
    )
    a = GazeTracking(ed, pd)
    cameras_dump = FileSource(SourceMode.DUMP, "../cameras_dump.json", True)
    gaze_dump = FileSource(SourceMode.DUMP, "../gaze_dump.json", True)

    a.compute_world_data(False, cameras_dump, gaze_dump, 2)
