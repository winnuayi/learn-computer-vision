import cv2
from keras.models import load_model
import imutils
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import numpy as np


WEBCAM = 0

THRESHOLD_CONFIDENCE = 0.40

FONT_SCALE = 1.0
GREEN = (0, 255, 0)
THICKNESS = 2

cap = cv2.VideoCapture(WEBCAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

PATH = 'models/mymodels/handwriting10.model'

model = load_model(PATH)

label_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
label_names = [l for l in label_names]

while True:
    # Capture frame-by-frame
    # if frame is read correctly ret is True
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # image = cv2.imread(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 30, 150)

    contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue

    # contours = imutils.grab_contours(contours)
    # contours = sort_contours(contours, method='left-to-right')[0]

    chars = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 100) and (h >= 15 and h <= 120):
            # get a part of image (which is the characters)
            roi = gray[y:y+h, x:x+w]

            # apply threshold
            ret, thresh = cv2.threshold(roi, 0, 255,
                                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            (t_h, t_w) = thresh.shape
            if t_w > t_h:
                thresh = imutils.resize(thresh, width=32)
            else:
                thresh = imutils.resize(thresh, height=32)

            (t_h, t_w) = thresh.shape
            d_x = int(max(0, 32 - t_w) / 2.0)
            d_y = int(max(0, 32 - t_h) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=d_y, bottom=d_y, left=d_x,
                                        right=d_x, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            padded = padded.astype('float32') / 255.0
            padded = np.expand_dims(padded, axis=-1)

            chars.append((padded, (x, y, w, h)))


    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype='float32')

    try:
        preds = model.predict(chars)
    except ValueError:
        continue

    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # pred is an array of 36 columns. Filled with probability of characters
        # from 0, 1, 2, ..., A, B, C, ..., X, Y, Z
        i = np.argmax(pred)

        prob = pred[i]
        label = label_names[i]

        if prob < THRESHOLD_CONFIDENCE:
            continue

        # print(pred)
        # print(i, prob, label)

        print("[INFO] {} - {:.2f}%".format(label, prob*100))
        cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, THICKNESS)
        cv2.putText(frame, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, GREEN, THICKNESS)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()