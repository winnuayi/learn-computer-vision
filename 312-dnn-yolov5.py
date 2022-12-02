import matplotlib.pyplot as plt
import numpy as np
import cv2


def cv2_imshow(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)


img = cv2.imread('images/people/elon.jpg')
cv2_imshow(img)

# model = 'models/yolov5n.onnx'
# net = cv2.dnn_Net(model)

cv2.waitKey(0)