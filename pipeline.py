from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd


class detectPipeline():
    def __init__(self) -> None:
        self.model = YOLO('Models/yolo_v8_nano_model.pt')
        self.class_names = {i: chr(65 + i) for i in range(26)}


    def detect_signs(self, img_path: str):
        # Data Preprocessing
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)

        # Making detections using YOLOv8 Nano
        detections = self.model(img_array)[0]
        sign_detections = []
        for sign in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = sign
            sign_detections.append([int(x1), int(y1), int(x2), int(y2), score, int(class_id)])
        print(sign_detections)
        return sign_detections

    def drawDetections2Image(self, img_path, detections):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        for bbox in detections:
            x1, y1, x2, y2, score, class_id = bbox
            cv.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
            cv.putText(img, text=f'{self.class_names[class_id]} ({round(score*100, 2)}%)', org=(x1, y1-20), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
                        color=(0, 0, 255), lineType=cv.LINE_AA, thickness=4)
        img_detections = np.array(img)
        return img_detections

    # get sign_detetction
    def extractTextResults(self, detections):
        text_results = ''
        for bbox in detections:
            x1, y1, x2, y2, score, class_id = bbox
            text_results += f'{self.class_names[class_id]} : {score}\n'
        return text_results