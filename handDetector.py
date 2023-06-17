import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import handLandmarks
import numpy as np
import cv2

folder = "C:\\Users\\Ani\\Desktop\\detector\\"  # 替换文件夹路径
file_name = os.listdir(folder)
model_asset_path=os.getcwd() + '\hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path)

options = vision.HandLandmarkerOptions(base_options=base_options, min_hand_detection_confidence=0.01, num_hands=2,
                                       min_hand_presence_confidence=0.01)
detector = vision.HandLandmarker.create_from_options(options)

for image_name in file_name:
    image_path = folder + image_name

    input_image = cv2.imread(image_path)
    balck_image = np.zeros(input_image.shape, np.uint8)
    balck_image[:] = [0, 0, 0]
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)
    annotated_image = handLandmarks.draw_landmarks_on_image(balck_image, detection_result)
    cv2.imwrite( folder + 'hand_' + image_name, annotated_image)
