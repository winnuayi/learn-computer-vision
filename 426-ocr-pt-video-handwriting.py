import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50


WEBCAM = 0
PATH = 'images/handwritten-numbers.jpg'

MODEL = 'resnet50.pth'

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
THICKNESS = 2
FONT_SCALE = 1
RESNET_SIZE = 32 # pixel
BORDER_SIZE = 20


def run_webcam():
    cap = cv2.VideoCapture(WEBCAM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        # if frame is read correctly ret is True
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = process(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def process(image):
    final_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 100, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for contour in contours:
        # extract x, y, width, height
        (x, y, w, h) = cv2.boundingRect(contour)

        if (w < 5 or w > 100):
            continue

        if (h < 10 or h > 120):
            continue

        roi = dilate[y:y+h, x:x+w]

        padded = cv2.copyMakeBorder(
            roi, top=BORDER_SIZE, bottom=BORDER_SIZE, left=BORDER_SIZE,
            right=BORDER_SIZE, borderType=cv2.BORDER_CONSTANT, value=BLACK)

        padded = cv2.resize(padded, (RESNET_SIZE, RESNET_SIZE))

        chars.append((padded, (x, y, w, h)))


    model = resnet50(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)

    model.load_state_dict(torch.load(MODEL))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with torch.no_grad():
        for i in range(len(chars)):
            roi = chars[i][0]
            (x, y, w, h) = chars[i][1]
            roi_tensor = transform(roi).unsqueeze(0)

            prediction = model(roi_tensor)

            _, predicted = torch.max(prediction.data, 1)

            result = str(predicted.item())

            cv2.rectangle(final_image, (x, y), (x+w, y+h), GREEN, THICKNESS)
            cv2.putText(final_image, result, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE, GREEN, THICKNESS)

    return final_image

run_webcam()