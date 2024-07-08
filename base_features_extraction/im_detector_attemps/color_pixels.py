import cv2
import numpy as np


def extract_yellow_pixels(image_path):
    # Load the image
    image = cv2.imread(image_path)
    print(image.shape)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the image to extract yellow pixels
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Apply the mask to the original image
    yellow_pixels = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Display the yellow pixels
    cv2.imshow("Yellow Pixels", yellow_pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = "data/frames_examples/frame_81.jpg"
extract_yellow_pixels(image_path)

"""import cv2
import numpy as np

# Load the image
image = cv2.imread("data/frames_examples/frame_81.jpg")

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Threshold the image to get only yellow pixels
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

# Bitwise-AND the original image with the yellow mask to get the yellow pixels
yellow_pixels = cv2.bitwise_and(image, image, mask=yellow_mask)

# join the red and yellow pixels
# yellow_red_pixels = cv2.bitwise_or(yellow_pixels, red_pixels)

# Display the yellow pixels
cv2.imshow("Yellow Pixels", yellow_pixels)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
