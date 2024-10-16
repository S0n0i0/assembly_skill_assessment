from __future__ import annotations
import cv2
from datetime import datetime

from utils.enums import LogCode, SourceMode, SensorMode, DisplayLevel, PathType
from utils.functions import compute_distance


class LogManager:
    """Class responsible for managing operation logs

    Attributes:
            show (bool): if True, logs are printed during operations.
            log_file (TextIOWrapper): if not None, logs are saved here.
            verbose (bool): if True, logs are detailed.
    """

    def __init__(
        self,
        display_level: DisplayLevel = DisplayLevel.LOW,
        path: str | None = None,
        new_file=False,
        verbose=True,
    ):
        """Initialize the LogManager object.

        Args:
            display_level (DisplayLevel): The display level for logs. Defaults to DisplayLevel.LOW.
            path (str | None): The path to the log file. Defaults to None.
            new_file (bool): If True, clears an existing log file. Defaults to False.
            verbose (bool): If True, logs are detailed. Defaults to True.
        """

        self.verbose = verbose
        self.display_level = display_level

        if path is None:
            self.log_file = None
            return
        try:
            self.log_file = open(path, "w" if new_file else "a")
        except:
            self.log_file = None
            self.path = None

    def log(
        self,
        source: str,
        code: LogCode,
        sub_code: int | None = None,
        message: str | None = None,
        display_level: DisplayLevel = DisplayLevel.HIGH,
    ):
        """Register the log message

        Args:
            source (str): The log source.
            code (LogCode): The log code.
            sub_code (int): The log subcode. Defaults to None.
            message (str): The log specification message. Defaults to None.
            display_level (DisplayLevel): The display level for the log. Defaults to DisplayLevel.HIGH.
        """

        now = datetime.now()
        sub_code_str = "_" + str(sub_code) if sub_code is not None else ""
        log_message = f"{source} - {code.value}{sub_code_str}"

        if self.verbose:
            log_message = f"{now} - {log_message}"
            log_message += " - " + message if message is not None else ""
        else:
            now = now.strftime("%d/%m/%Y %H:%M:%S")
            log_message = f"{now} - {log_message}"

        if (
            self.display_level == DisplayLevel.ONLY_ERRORS
            and code == LogCode.ERROR
            or display_level.value <= self.display_level.value
        ):
            print(log_message)

        if self.log_file is not None:
            self.log_file.write(f"{log_message}\n")


class Source:
    """Class responsible for storing data source information.

    Attributes:
            path (str): file path where the data source is stored
    """

    def __init__(self, mode: SourceMode, params=None, new_dump=True):
        """Initialize the Source object.

        Args:
            mode (SourceMode): source mode
        """

        self.mode = mode
        self.new_dump = new_dump
        self.params = params

    @staticmethod
    def compare(source1, source2):
        """Compare two sources.

        Args:
            source1 (Source): first source to compare
            source2 (Source): second source to compare

        Returns:
            bool: True if sources are equal, False otherwise
        """
        return source1 and source2 and source1.mode == source2.mode


class PathSource(Source):
    """Class responsible for storing data source information.

    Attributes:
            path (str): file path where the data source is stored
            new_file (bool): if Source is used for saving data, specify if an eventual old file is cleared or not
    """

    def __init__(
        self,
        mode: SourceMode,
        path: str | None = None,
        new_dump=True,
        params=None,
        type: PathType = PathType.FILE,
    ):
        """Initialize the Source object.

        Args:
            mode (SourceMode): source mode
            path (str): file path where the data source is stored
        """

        super().__init__(mode, params, new_dump)
        self.path = path
        self.type = type

    @staticmethod
    def compare(source1, source2):
        """Compare two sources.

        Args:
            source1 (FileSource): first source to compare
            source2 (FileSource): second source to compare

        Returns:
            bool: True if sources are equal, False otherwise
        """
        return Source.compare(source1, source2) and source1.path == source2.path


class DataSource(Source):
    def __init__(self, mode: SourceMode, data=None, params=None, new_dump=True):
        super().__init__(mode, params, new_dump)
        self.data = data


class ChannelHandler:

    def __init__(
        self, mode: SensorMode | None = None, data=None, ref=None, new_dump=None
    ) -> None:
        self.data = data
        self.ref = ref
        self.mode = mode
        self.new_dump = new_dump

    def get_dump_path(self):
        if isinstance(self.new_dump, str):
            return self.new_dump
        elif self.new_dump == True:
            return self.ref
        else:
            return None


class VideoReader:
    """
    A class for reading video files and extracting frames.

    Args:
        video_path (str): The path to the video file.
        frame (int, optional): The starting frame index (default is 0).
        grey_scale (bool, optional): Whether to convert frames to grayscale (default is True).

    Attributes:
        video: The OpenCV VideoCapture object.
        path (str): The path to the video file.
        grey_scale (bool): Whether frames are converted to grayscale.

    Methods:
        __iter__(): Returns the iterator object.
        __next__(): Returns the next frame in the video.
        __del__(): Releases the video capture object.
        read(): Reads the next frame from the video.
        release(): Releases the video capture object.
        get_index_frame(int_value=True): Returns the current frame index.
        previous_frame(): Moves to the previous frame.
        set_frame(frame): Sets the current frame index.
        get_frame(frame=None): Returns the specified frame.

    """

    def __init__(self, video_path, frame=0, grey_scale=True):
        self.video = cv2.VideoCapture(video_path)
        self.path = video_path

        if not self.video.isOpened():
            print("Impossibile aprire il video:", video_path)
            self.video = None
        else:
            self.grey_scale = grey_scale
            if frame > 0:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def __iter__(self):
        return self

    def __next__(self):
        if self.video is None:
            return None
        else:
            ret, frame = self.read()
            if not ret:
                raise StopIteration
            return frame

    def __del__(self):
        if self.video is not None:
            self.video.release()

    def read(self):
        if self.video is None:
            return False, None

        ret, frame = self.video.read()

        if ret and self.grey_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return ret, frame

    def release(self):
        if self.video is not None:
            self.video.release()

    def get_index_frame(self, int_value=True):
        return (
            int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            if int_value
            else self.video.get(cv2.CAP_PROP_POS_FRAMES)
        )

    def previous_frame(self):
        return self.video.set(cv2.CAP_PROP_POS_FRAMES, self.get_index_frame() - 1)

    def set_frame(self, frame):
        return self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def get_frame(self, frame=None):
        """
        Returns the specified frame from the video.

        Args:
            frame (int, optional): The frame index to retrieve (default is None, which returns the next frame).

        Returns:
            image: The frame image.

        """
        if self.video is None:
            return None
        elif frame is None:
            ret, image = self.read()
        elif frame > 0:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, image = self.read()

            # Convert the frame to grayscale
            if ret and self.grey_scale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return None

        return image


class BoundingBox:
    """
    Represents a bounding box in an image.

    Attributes:
        x (float): The x-coordinate of the top-left corner of the bounding box.
        y (float): The y-coordinate of the top-left corner of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
    """

    def __init__(self, array: list[float, float, float, float], is_xywh=True):
        """
        Initializes a BoundingBox object.

        Args:
            array (list[float, float, float, float]): The array representing the bounding box.
                If `is_xywh` is True, the array should be in the format [x, y, w, h].
                If `is_xywh` is False, the array should be in the format [x1, y1, x2, y2].
            is_xywh (bool, optional): Indicates whether the array is in the format [x, y, w, h].
                Defaults to True.
        """
        if is_xywh:
            # array is [x, y, w, h]
            self.x = array[0]
            self.y = array[1]
            self.w = array[2]
            self.h = array[3]
        else:
            # array is [x1, y1, x2, y2]
            self.x = array[0]
            self.y = array[1]
            self.w = array[2] - array[0]
            self.h = array[3] - array[1]

    def to_xywh(self):
        """
        Converts the bounding box to the format [x, y, w, h].

        Returns:
            list[float, float, float, float]: The bounding box in the format [x, y, w, h].
        """
        return [self.x, self.y, self.w, self.h]

    def to_xyxy(self):
        """
        Converts the bounding box to the format [x1, y1, x2, y2].

        Returns:
            list[float, float, float, float]: The bounding box in the format [x1, y1, x2, y2].
        """
        return [self.x, self.y, self.x + self.w, self.y + self.h]

    def get_area(self):
        """
        Calculates the area of the bounding box.

        Returns:
            float: The area of the bounding box.
        """
        return self.w * self.h

    def get_center(self):
        """
        Calculates the center coordinates of the bounding box.

        Returns:
            tuple[float, float]: The center coordinates of the bounding box.
        """
        return self.x + self.w // 2, self.y + self.h // 2

    def get_intersection(self, other: BoundingBox):
        """
        Calculates the intersection bounding box between the current bounding box and another bounding box.

        Args:
            other (BoundingBox): The other bounding box.

        Returns:
            BoundingBox: The intersection bounding box.
        """
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

    def get_union(self, other: BoundingBox):
        """
        Calculates the union area between the current bounding box and another bounding box.

        Args:
            other (BoundingBox): The other bounding box.

        Returns:
            float: The union area.
        """
        inter = self.get_intersection(other)
        return self.get_area() + other.get_area() - inter.get_area()

    def get_iou(self, other: BoundingBox):
        """
        Calculates the Intersection over Union (IoU) between the current bounding box and another bounding box.

        Args:
            other (BoundingBox): The other bounding box.

        Returns:
            float: The IoU value.
        """
        inter = self.get_intersection(other)
        union = self.get_union(other)
        return inter.get_area() / union

    def get_distance(self, point: tuple[int, int]):
        """
        Calculates the Euclidean distance between the center of the bounding box and a given point.

        Args:
            point (tuple[int, int]): The coordinates of the point.

        Returns:
            float: The Euclidean distance.
        """
        center = self.get_center()
        return compute_distance(center, point)


class Sequence:
    """
    Represents an Assembly101 sequence (i.e. assembly-disassembly record)
    """

    def __init__(self, sequence: str, coarse_title=False) -> None:
        """
        Initializes a Sequence object.

        Args:
            sequence (str): string which represents the sequence. It can also be a split row.
            coarse_title (bool, optional): represents if the string passed is a title of a coarse labels file
        """
        self.coarse_title = coarse_title
        tmp_sequence = sequence.strip().split("_")
        if (not self.coarse_title and len(tmp_sequence) >= 9) or (
            coarse_title and len(tmp_sequence) >= 10
        ):
            self.sequence_list = sequence.split("_")
        else:
            print("The provided string is not a Sequence")
            raise

    def __str__(self) -> str:
        full_sequence = (
            "_".join(self.sequence_list)
            if not self.coarse_title
            else "_".join(self.sequence_list[1:])
        )
        if "," in full_sequence:
            full_sequence = full_sequence.split(",")[1]
        if "/" in full_sequence:
            full_sequence = full_sequence.split("/")[0]

        return full_sequence

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: Sequence | str):
        if isinstance(other, Sequence):
            return str(self) == str(other)
        elif isinstance(other, str):
            return str(self) == other
        else:
            return False

    def __add__(self, other: Sequence | str):
        if isinstance(other, Sequence):
            return Sequence(str(self) + str(other))
        elif isinstance(other, str):
            return Sequence(str(self) + other)
        else:
            return None

    @property
    def person(self):
        return self.sequence_list[4 if not self.coarse_title else 5]

    @property
    def toy(self):
        return self.sequence_list[3 if not self.coarse_title else 4].split("-")[1]
