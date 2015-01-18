import cv2
import numpy as np

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

cv2.namedWindow("preview", cv2.WND_PROP_FULLSCREEN)
vc = cv2.VideoCapture(1)

x_offset=y_offset=250
black = (0, 0, 0)

width, height = 1366, 768
l_img = create_blank(width, height, rgb_color=black)


if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False

while rval:
	combined = np.append(frame, frame, axis=1)
	cv2.setWindowProperty("preview", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

	combined = cv2.resize(combined, (0,0), fx=0.4, fy=0.4)
	l_img[y_offset:y_offset+combined.shape[0], x_offset:x_offset+combined.shape[1]] = combined

	cv2.imshow("preview", l_img)
	rval, frame = vc.read()

	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
		break
cv2.destroyWindow("preview")
