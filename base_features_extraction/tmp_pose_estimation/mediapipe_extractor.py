import cv2
from enum import Enum
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from utils.classes import Source, FileSource
from utils.constants import log_manager
from utils.enums import LogCode, SensorMode, SourceMode
from utils.functions import load_dump


class MediaPipeOpCode(Enum):
    """Enum class representing mediapipe Operation Codes in logs."""

    INPUT_DUMP = 0
    ATTRIBUTES_LOADING = 1

    # class MediapipeAttributes:
    """
    Class representing the attributes of a Mediapipe model.

    Attributes:
        model_asset_path (str): Model asset path given in input to task_options class (e.g., "/path/to/model.tflite").
        base_options: The base options for the task (e.g., mp.tasks.BaseOptions).
        vision_task: The vision task (e.g., vision.ObjectDetector).
        vision_task_options: The options for the vision task (e.g., vision.ObjectDetectorOptions).
        task_result: The result of the task (optional, e.g., (mp.tasks.components.containers.detections.DetectionResult)).

    Methods:
        __init__: Initialize the MediapipeAttributes object.
    """

    """def __init__(
        self,
        model_asset_path: str,
        base_options,
        vision_task,
        vision_task_options,
        task_result=None,
    ):"""
    """Initialize the MediapipeAttributes object.

        Args:
            model_asset_path (str): tflite model asset path given in input to task_options class.
            base_options: The base options for the task (e.g., mp.tasks.BaseOptions).
            vision_task: The vision task (e.g., vision.ObjectDetector).
            vision_task_options: The options for the vision task (e.g., vision.ObjectDetectorOptions).
            task_result: The result of the task (optional, e.g., (mp.tasks.components.containers.detections.DetectionResult)).
        """
    """self.model_asset_path = model_asset_path
        self.base_options = base_options
        self.vision_task = vision_task
        self.vision_task_options = vision_task_options
        self.task_result = task_result"""


class MediapipeSource:
    """Class responsible for storing mediapipe data source information."""

    def __init__(
        self,
        source: Source = None,
        model: FileSource = None,
    ):
        """Initialize the Source object."""
        self.source = source
        self.model = model


class MediapipeExtractor:
    """Class responsible for object detection and pose landmark detection using Mediapipe."""

    def __init__(
        self,
        object_source: MediapipeSource = None,
        pose_source: MediapipeSource = None,
        hand_source: MediapipeSource = None,
    ):
        BaseOptions = mp.tasks.BaseOptions
        default = True
        if object_source and isinstance(object_source.source, FileSource):
            self.object_source = object_source.source
            if object_source.source.mode == SourceMode.DUMP:
                try:
                    self.object_data = load_dump(self.object_source.path)
                    self.object_mode = SensorMode.OFFLINE_DUMP
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        MediaPipeOpCode.INPUT_DUMP.value,
                        "Dumped object data loaded successfully",
                    )
                    default = False
                except:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        MediaPipeOpCode.INPUT_DUMP.value,
                        "Error loading dumped object data",
                    )
            elif object_source.source.mode == SourceMode.VIDEO and object_source.model:
                ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
                self.object_detector_options = ObjectDetectorOptions(
                    base_options=BaseOptions(model_asset_path=object_source.model.path),
                    max_results=3,
                    category_denylist=["person", "surfboard"],
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                )
                self.object_detector = (
                    mp.tasks.vision.ObjectDetector.create_from_options(
                        self.object_detector_options
                    )
                )
                self.object_detector_result = None
                self.object_mode = SensorMode.OFFLINE
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    MediaPipeOpCode.ATTRIBUTES_LOADING.value,
                    "Mediapipe object detectore attributes set successfully",
                )
                default = False

        if default:
            self.object_mode = None
            self.object_source = None
            self.object_data = {}
            self.object_detector = None
            self.object_detector_result = None
            self.object_detector_options = None

        default = True
        if pose_source and isinstance(pose_source.source, FileSource):
            self.pose_source = pose_source.source
            if pose_source.source.mode == SensorMode.OFFLINE_DUMP:
                try:
                    self.pose_data = load_dump(self.pose_source.path)
                    self.pose_mode = SensorMode.OFFLINE_DUMP
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        MediaPipeOpCode.INPUT_DUMP.value,
                        "Dumped pose data loaded successfully",
                    )
                    default = False
                except:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        MediaPipeOpCode.INPUT_DUMP.value,
                        "Error loading dumped pose data",
                    )
            elif pose_source.source.mode == SourceMode.VIDEO and pose_source.model:
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                self.pose_landmarker_options = PoseLandmarkerOptions(
                    BaseOptions(model_asset_path=pose_source.model.path),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                )
                self.pose_landmarker = (
                    mp.tasks.vision.PoseLandmarker.create_from_options(
                        self.pose_landmarker_options
                    )
                )
                self.pose_landmarker_result = None
                self.pose_mode = SensorMode.OFFLINE
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    MediaPipeOpCode.ATTRIBUTES_LOADING.value,
                    "Mediapipe pose landmarker attributes set successfully",
                )
                default = False

        if default:
            self.pose_mode = None
            self.pose_source = None
            self.pose_data = {}
            self.pose_landmarker = None
            self.pose_landmarker_result = None
            self.pose_landmarker_options = None

        default = True
        if hand_source and isinstance(hand_source.source, FileSource):
            self.hand_source = hand_source.source
            if hand_source.source.mode == SensorMode.OFFLINE_DUMP:
                try:
                    self.hand_data = load_dump(self.hand_data.path)
                    self.hand_mode = SensorMode.OFFLINE_DUMP
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.SUCCESS,
                        MediaPipeOpCode.INPUT_DUMP.value,
                        "Dumped hand data loaded successfully",
                    )
                    default = False
                except:
                    log_manager.log(
                        self.__class__.__name__,
                        LogCode.ERROR,
                        MediaPipeOpCode.INPUT_DUMP.value,
                        "Error loading dumped hand data",
                    )
            elif hand_source.source.mode == SourceMode.VIDEO and hand_source.model:
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                self.hand_landmarker_options = HandLandmarkerOptions(
                    BaseOptions(model_asset_path=hand_source.model.path),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                self.hand_landmarker = (
                    mp.tasks.vision.HandLandmarker.create_from_options(
                        self.hand_landmarker_options
                    )
                )
                self.hand_landmarker_result = None
                self.hand_mode = SensorMode.OFFLINE
                log_manager.log(
                    self.__class__.__name__,
                    LogCode.SUCCESS,
                    MediaPipeOpCode.ATTRIBUTES_LOADING.value,
                    "Mediapipe hand landmarker attributes set successfully",
                )
                default = False

        if default:
            self.hand_mode = None
            self.hand_source = None
            self.hand_data = {}
            self.hand_landmarker = None
            self.hand_landmarker_result = None
            self.hand_landmarker_options = None

        if FileSource.compare(self.object_source, self.pose_source):
            self.same_source = True
        else:
            self.same_source = False

    def print_result(self, result, output_image: mp.Image, timestamp_ms: int):
        """
        Print the result of the task.

        Args:
            result: The result of the task.
            output_image: The output image.
            timestamp_ms: The timestamp in milliseconds.

        """
        if isinstance(result, self.object_detector_result):
            print("detection result: {}".format(result))
        else:
            print("result is not a detection result")


if __name__ == "__main__":
    # check arguments
    compute_object = False
    compute_pose = True
    compute_hand = False

    video_source = FileSource(
        SourceMode.VIDEO,
        "./data/video_examples/C10118_rgb.mp4",
    )
    object_source = MediapipeSource(
        video_source,
        FileSource(
            SourceMode.DUMP,
            "./models/mediapipe/efficientdet_lite2.tflite",
        ),
    )
    pose_source = MediapipeSource(
        video_source,
        FileSource(
            SourceMode.DUMP,
            "./models/mediapipe/pose_landmarker_full.task",
        ),
    )
    hand_source = MediapipeSource(
        video_source,
        FileSource(
            SourceMode.DUMP,
            "./models/mediapipe/hand_landmarker.task",
        ),
    )
    mediapipe_extractor = MediapipeExtractor(
        object_source if compute_object else None,
        pose_source if compute_pose else None,
        hand_source if compute_hand else None,
    )
    mp_drawing = mp.solutions.drawing_utils

    # Use OpenCV’s VideoCapture to load the input video.
    video = cv2.VideoCapture(video_source.path)
    # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    # Loop through each frame in the monochrome video using VideoCapture#read()
    while video.isOpened():
        # Read the frame using VideoCapture#read()
        ret, frame = video.read()
        # crop the left side of the image
        """height, width, _ = frame.shape
        frame = frame[int(height / 2) : height, int(width / 2) : width]"""

        # crop frame to get upper left corner
        # Break the loop if the frame is empty.
        if not ret:
            break

        # crop frame to get upper left corner in 360p
        # frame = frame[160:270, :640]
        # rotate frame
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Convert the frame to RGB using cv2.cvtColor()

        # Convert the frame to MediaPipe’s Image object.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame.astype("uint8")
        )

        # Calculate the timestamp of the current frame
        frame_timestamp_ms = int(1000 * frame_index / fps)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            cv2.imwrite(f"./data/frames_examples/frame_{frame_index}.jpg", frame)

        if compute_object:
            # Perform object detection on the video frame.
            detection_result = mediapipe_extractor.object_detector.detect_for_video(
                mp_image, frame_timestamp_ms
            )

            if detection_result.detections:
                print("Detected objects:")
                for detection in detection_result.detections:
                    print(detection.categories[0].category_name)
                    bbox = detection.bounding_box
                    cv2.rectangle(
                        frame,
                        (int(bbox.origin_x), int(bbox.origin_y)),
                        (
                            int(bbox.origin_x + bbox.width),
                            int(bbox.origin_y + bbox.height),
                        ),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        detection.categories[0].category_name,
                        (int(bbox.origin_x), int(bbox.origin_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        if compute_pose:
            # Perform pose landmark detection on the video frame.
            pose_landmarks = mediapipe_extractor.pose_landmarker.detect_for_video(
                mp_image, frame_timestamp_ms
            )

            pose_landmarks_list = pose_landmarks.pose_landmarks
            # pose_landmarks_list = pose_landmarks.hand_landmarks
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in pose_landmarks
                    ]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )

        if compute_hand:
            # Perform hand landmark detection on the video frame.
            hand_landmarks = mediapipe_extractor.hand_landmarker.detect_for_video(
                mp_image, frame_timestamp_ms
            )

            hand_landmarks_list = hand_landmarks.hand_landmarks
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in hand_landmarks
                    ]
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                )

        # Mostra il frame con i risultati
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    video.release()
    cv2.destroyAllWindows()
