import cv2
import numpy as np

bright = cv2.imread("../Image/18.JPG")
dark = cv2.imread("../Image/18.JPG")

bright_lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
dark_lab = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)

bright_ycb = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
dark_ycb = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)

bright_hsv = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
dark_hsv = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)

# python
bgr_red = [40, 40, 200]
bgr_light_blue = [176, 181, 57]
bgr_yellow = [98, 158, 195]
bgr_white = [230, 230, 230]
bgr = bgr_white
thresh = 40

min_bgr = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
max_bgr = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

mask_bgr = cv2.inRange(bright, min_bgr, max_bgr)
result_bgr = cv2.bitwise_and(bright, bright, mask=mask_bgr)

# convert 1D array to 3D, then convert it to HSV and take the first element
# this will be same as shown in the above figure [65, 229, 158]
hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

min_hsv = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
max_hsv = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

mask_hsv = cv2.inRange(bright_hsv, min_hsv, max_hsv)
result_hsv = cv2.bitwise_and(bright_hsv, bright_hsv, mask=mask_hsv)

# convert 1D array to 3D, then convert it to YCrCb and take the first element
ycb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]

min_ycb = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
max_ycb = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])

mask_ycb = cv2.inRange(bright_ycb, min_ycb, max_ycb)
result_ydb = cv2.bitwise_and(bright_ycb, bright_ycb, mask=mask_ycb)

# convert 1D array to 3D, then convert it to LAB and take the first element
lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

min_lab = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
max_lab = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

mask_lab = cv2.inRange(bright_lab, min_lab, max_lab)
result_lab = cv2.bitwise_and(bright_lab, bright_lab, mask=mask_lab)

cv2.imshow("Result BGR", result_bgr)
cv2.waitKey(0)
cv2.imshow("Result HSV", result_hsv)
cv2.waitKey(0)
cv2.imshow("Result YCB", result_ydb)
cv2.waitKey(0)
cv2.imshow("Output LAB", result_lab)
cv2.waitKey(0)
cv2.destroyAllWindows()

detect_img_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
_, detect_thresh_img = cv2.threshold(detect_img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Result BGR", detect_img_gray)
cv2.waitKey(0)
cv2.imshow("Result HSV", detect_thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
