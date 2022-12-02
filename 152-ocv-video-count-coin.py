# 1. hidupkan rtmp server menggunakan repo rtsp-simple-server
# 2. broadcast live streaming menggunakan prism live mobile app
# 3. jalankan script ini
import cv2
import numpy as np


PATH = "rtmp://192.168.100.95:1935/awesome/awesome"

cap = cv2.VideoCapture(PATH)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    canny = cv2.Canny(blur, 10, 240)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel, iterations=1)
    (cnt, hierarchy) = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

    cv2.putText(rgb, f"Coin: {len(cnt)}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, (0, 255, 0), 5)

    cv2.imshow("coin", rgb)

    if cv2.waitKey(1) == ord('q'):
        break