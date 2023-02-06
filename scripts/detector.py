import rospkg

import cv2 as cv
import numpy as np
import os

import time

rospack = rospkg.RosPack()

kuav_simulation_path = rospack.get_path("kuav_simulation")
kuav_detection_path = rospack.get_path("kuav_detection")


model = cv.dnn.readNetFromONNX(os.path.join(kuav_detection_path, "resources/yolov7_256x320.onnx"))
ln = model.getUnconnectedOutLayersNames()

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture(os.path.join(kuav_simulation_path, "rosbag/video_out/2023-01-29-21-03-43-fwd-cam.mp4"))
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) / 3)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) / 3)

classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

while cap.isOpened():
    start_time = time.perf_counter()
    ret, frame_rs = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    rows, cols, channels = frame_rs.shape
    # print(rows)
    blob = cv.dnn.blobFromImage(frame_rs, 1/530, (320, 256), swapRB=True, mean=(0,0,0), crop= False) #, mean=(104, 117, 123))
    model.setInput(blob)
    l_outputs = model.forward(ln)

    out= l_outputs[0]
    n_detections= out.shape[1]
    height, width= frame_rs.shape[:2]

    x_scale= width/320
    y_scale= height/256
    conf_threshold= 0.7
    score_threshold= 0.5
    nms_threshold= 0.5

    class_ids=[]
    score=[]
    boxes=[]


    for i in range(n_detections):
        detect=out[0][i]
        confidence= detect[4]
        if confidence >= conf_threshold:
            class_score= detect[5:]
            class_id= np.argmax(class_score)
            if (class_score[class_id]> score_threshold):
                score.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                left= int((x - w/2)* x_scale )
                top= int((y - h/2)*y_scale)
                width = int(w * x_scale)
                height = int( y*y_scale)
                box= np.array([left, top, width, height])
                boxes.append(box)

    indices = cv.dnn.NMSBoxes(boxes, np.array(score), conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        left, top, width, height = box[0:4]
        cv.rectangle(frame_rs, (left, top), (left + width, top + height), (0, 0, 255), 3)
        label = "{}:{:.2f}".format(classes[class_ids[i]], score[i])
        text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        dim, baseline = text_size[0], text_size[1]
        cv.rectangle(frame_rs, (left, top), (left + dim[0], top + dim[1] + baseline), (0,0,0), cv.FILLED)
        cv.putText(frame_rs, label, (left, top + dim[1]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv.LINE_AA)

    cv.imshow("frame", frame_rs)
    dt = time.perf_counter() - start_time
    print(f"FPS: {1/dt} dt: {dt}")
    key = cv.waitKey(1)
    if key == "q":
        break