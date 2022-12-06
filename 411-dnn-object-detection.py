import cv2


SRC = "rtmp://192.168.100.95:1935/awesome/awesome"

cap = cv2.VideoCapture(SRC)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Labels of network.
    class_names = { 0: 'background',
        1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
        10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
        14: 'motorbike', 15: 'person', 16: 'pottedplant',
        17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

    # config_path = 'models/object-detection/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config'
    # weight_path = 'models/object-detection/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/saved_model.pb'

    config_path = 'models/object-detection/MobileNetSSD_deploy.caffemodel'
    weight_path = 'models/object-detection/MobileNetSSD_deploy.prototxt'

    net = cv2.dnn_DetectionModel(weight_path, config_path)
    net.setInputSize(300, 300)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    class_ids, confs, bbox = net.detect(frame, confThreshold=0.5)
    if len(class_ids) > 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
            conf_str = '{0:.2f}'.format(confidence)
            cv2.putText(frame, f"{class_names[class_id]} {conf_str}", (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("output", frame)

    if cv2.waitKey(1) == ord('q'):
        break