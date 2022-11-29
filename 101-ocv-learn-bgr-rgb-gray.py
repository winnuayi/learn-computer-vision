import cv2
from matplotlib import pyplot as plt

IMAGE_FILE = "image.jpg"

img = cv2.imread(IMAGE_FILE)
cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("rgb", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()