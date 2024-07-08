from mmpose.apis import MMPoseInferencer, visualize
import cv2
import numpy as np

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer("wholebody")

video_path = "data/video_examples/C10118_rgb.mp4"
cap = cv2.VideoCapture(video_path)

# first_frame = True

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        # The MMPoseInferencer API employs a lazy inference approach,
        # creating a prediction generator when given input
        result_generator = inferencer(frame, show=True, wait_time=1)
        result = next(result_generator)

        # result["predictions"][0].sort(key=lambda x: x["bbox_score"], reverse=True)
        person = result["predictions"][0][0]

        keypoints = np.array([person["keypoints"]])
        keypoint_scores = np.array([person["keypoint_scores"]])
        """visualize(
            frame,
            keypoints,
            keypoint_scores,
            "models/mediapipe/mmpose/coco_wholebody.py",
            show=True,
        )"""

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        print("Ehm")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
