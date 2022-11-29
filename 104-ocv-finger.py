# BELUM SELESAI

import cv2
import os
import time

# konfigurasi dimensi video yang ditampilkan
w_cam, h_cam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

# list of filename gambar gesture tangan
PATH = 'images/fingers/gestures'
my_list = os.listdir(PATH)

# list of path gambar gesture tangan
overlay_list = []
for img_path in my_list:
    image = cv2.imread(f'{PATH}/{img_path}')
    print(f'{PATH}/{img_path}')
    overlay_list.append(image)

# untuk menghitung FPS
p_time = 0

while True:
    # baca video
    success, img = cap.read()

    # dapatkan ukuran height, weight, channel (3, RGB)
    h, w, c = overlay_list[0].shape

    # tampilkan gambar gesture tangan di video, di kiri atas
    img[0:h, 0:w] = overlay_list[0]

    # hitung FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # tampilkan informasi FPS di kanan atas
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)