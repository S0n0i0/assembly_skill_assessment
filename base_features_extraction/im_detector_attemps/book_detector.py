# import the necessary packages
import numpy as np
import cv2

# load the image, convert it to grayscale, and blur it
image = cv2.imread(
    "C:/Simone/Scuola/Universita/Magistrale/Secondo anno/Tesi/Project/assembly_skill_assessment/data/frames_examples/frame_24.jpg"
)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
gray = cv2.filter2D(gray, -1, sharpening_kernel)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
# crop the left side of the image
"""height, width = gray.shape
gray = gray[int(height / 2) : height, int(width / 2) : width]"""

"""# Applica il filtro di Sobel per rilevare i bordi
sobelx = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

# Combina i risultati di Sobel lungo gli assi X e Y
gray = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)"""
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# detect edges in the image
edged = cv2.Canny(gray, 10, 250)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# construct and apply a closing kernel to 'close' gaps between 'white'
# pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0)

# find contours (i.e. the 'outlines') in the image and initialize the
# total number of books found
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the approximated contour has four points, then assume that the
    # contour is a book -- a book is a rectangle and thus has four vertices
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        total += 1

# display the output
print("I found {0} books in that image".format(total))
cv2.imshow("Output", image)
cv2.waitKey(0)
