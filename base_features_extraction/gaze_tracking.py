import json
import numpy as np


class GazeTracking:
    def __init__(self, extrinsic_source, position_source):
        """Initialize the GazeTracking object.

        Args:
            extrinsic_source (str): Path to the extrinsic parameters JSON file.
            position_source (str): Path to the camera positions JSON file.

        Attributes:
            extrinsic_data (dict): Extrinsic parameters of the cameras.
            position_data (dict): Positions of the cameras.
            world_coordinates (dict): World coordinates of the cameras.

        Raises:
            FileNotFoundError: If the JSON files are not found.
        """
        try:
            self.load_data(extrinsic_source, position_source)
        except FileNotFoundError:
            print("File for offline analysis not found")

        self.world_coordinates = {}

    def load_data(self, extrinsic_path, position_path):
        """Load extrinsic and position data from JSON files.

        Args:
            extrinsic_path (str): Path to the extrinsic parameters JSON file.
            position_path (str): Path to the camera positions JSON file.
        """

        with open(extrinsic_path, "r") as f:
            self.extrinsic_data = json.load(f)
        with open(position_path, "r") as f:
            self.position_data = json.load(f)

    def set_world_coordinates(self, world_coordinates_source):
        """Set the world coordinates of the cameras.

        Args:
            world_coordinates_source (str): Path to the world coordinates JSON file.
        """

        with open(world_coordinates_source, "r") as f:
            self.world_coordinates = json.load(f)

    def get_world_coordinates(self, dump=False, force=False):
        """Calculate the world coordinates of the cameras.

        Args:
            dump (bool): If True, print the world coordinates.
            force (bool): If True, recalculate the world coordinates.

        Returns:
            dict: World coordinates of the cameras.
        """

        # If the world coordinates are already calculated and force is False, return them
        if self.world_coordinates and not force:
            return self.world_coordinates

        # Iterate over the frames and cameras to calculate the world coordinates
        for frame_id, cameras in self.position_data.items():
            self.world_coordinates[frame_id] = {}
            for camera_id, position in cameras.items():
                # Convert the position to a 4D vector with the fourth component set to 1
                position_4d = np.append(position, 1)

                # Get the corresponding extrinsic transformation matrix
                extrinsic_matrix = np.array(self.extrinsic_data[frame_id][camera_id])

                # Compute the derived coordinates by multiplying the matrix by the position vector
                self.world_coordinates[frame_id][camera_id] = np.dot(
                    extrinsic_matrix, position_4d
                ).tolist()

        # Dump in a json the world coordinates if requested
        if dump:
            json.dumps(self.world_coordinates, indent=4)

        return self.world_coordinates
