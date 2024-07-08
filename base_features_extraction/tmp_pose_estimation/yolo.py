import cv2
import mediapipe as mp
from ultralytics import YOLO
from mediapipe.framework.formats import landmark_pb2

# Load the YOLOv8 model
model = YOLO("yolov8n-pose.pt")  # load an official model
# model = YOLO("yolov8n.pt")

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hand_landmarker_options = HandLandmarkerOptions(
    BaseOptions(model_asset_path="./models/mediapipe/hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    hand_landmarker_options
)

# Open the video file
video_path = "data/video_examples/C10118_rgb.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.astype("uint8"))
    frame_timestamp_ms = int(1000 * frame_index / fps)

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        hand_landmarks = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

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
                annotated_frame,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            )

        frame_index += 1
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
