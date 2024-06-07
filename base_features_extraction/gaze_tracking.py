from enum import Enum
import json
import numpy as np
from utils.classes import Source
from utils.constants import log_manager
from utils.enums import LogCode, SourceMode, SensorMode
from utils.functions import load_dump


class GazeSubmode(Enum):
    OUTPUT_DUMP = "OUTPUT_DUMP"
    INPUT_DUMP = "INPUT_DUMP"


class GazeTracking:
    """Class responsible gaze tracking analysis.

    Attributes:
            extrinsic_data (dict): Extrinsic parameters of the cameras.
            position_data (dict): Positions of the cameras.
            world_coordinates (dict): World coordinates of the cameras.
    """

    def __init__(
        self,
        extrinsic_source: Source = None,
        position_source: Source = None,
        gaze_world_dump_path: str = None,
    ):
        """Initialize the GazeTracking object.

        Args:
            extrinsic_source (utils.classes.Source): Object containing information about the extrinsic parameters source.
            position_source (utils.classes.Source): Object containing information about the position parameters source.
            gaze_world_dump_path (str): Path of the coordinates of gaze w.r.t. the world.
            verbose (bool): if True, logs are printed during operations.
        """

        default = True

        if gaze_world_dump_path:
            try:
                self.gaze_world_coordinates = load_dump(gaze_world_dump_path)
                self.world_path = gaze_world_dump_path
                self.extrinsic_data = None
                self.position_data = None
                self.mode = SensorMode.OFFLINE_DUMP
                self.submode = GazeSubmode.OUTPUT_DUMP
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    0,
                    "Dumped gaze world coordinates loaded successfully",
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    0,
                    "Gaze world coordinates not loaded",
                )

        if default:
            self.world_path = None
            self.gaze_world_coordinates = None

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
                    1,
                    "Dumped cameras data loaded successfully",
                )
                return
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    1,
                    "Cameras data dump not loaded",
                )

        if default:
            self.extrinsic_path = None
            self.position_path = None
            self.extrinsic_data = {}
            self.position_data = {}

    def compute_world_coordinates(self, dump: str = None, new_file=True, force=False):
        """Calculate the world coordinates of the cameras.

        Args:
            dump (str): If specified, save the world coordinates in that JSON file.
            force (bool): If True, recalculate the world coordinates.

        Returns:
            dict[str, dict[str, list[list[float]]]]: World coordinates of the cameras.
        """

        # If force is False return the gaze world coordinates which are already calculated
        if self.gaze_world_coordinates and not force:
            log_manager.log(
                self.__class__.__name__, LogCode.SUCCESS, 2, "Old data returned"
            )
            return self.gaze_world_coordinates
        else:
            self.gaze_world_coordinates = {}

        # Iterate over the frames and cameras to calculate the world coordinates
        for frame_id, cameras in self.position_data.items():
            self.gaze_world_coordinates[frame_id] = {}
            for camera_id, position in cameras.items():
                # Convert the position to a 4D vector with the fourth component set to 1
                position_4d = np.append(position, 1)

                # Get the corresponding extrinsic transformation matrix
                extrinsic_matrix = np.array(self.extrinsic_data[frame_id][camera_id])

                # Compute the derived coordinates by multiplying the matrix by the position vector
                self.gaze_world_coordinates[frame_id][camera_id] = np.dot(
                    extrinsic_matrix, position_4d
                ).tolist()

        # Dump in a json the world coordinates if requested
        if dump is not None:
            try:
                with open(dump, "w" if new_file else "a") as f:
                    json.dump(self.gaze_world_coordinates, f)
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    3,
                    "Output gaze world coordinates dumped succesfully",
                )
            except:
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.ERROR,
                    3,
                    "Dump of gaze world coordinates failed",
                )

        return self.gaze_world_coordinates
