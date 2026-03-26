model_path = "D:/new_quant/models/hand_landmarker.task"
video_path = "D:/new_quant/data/videos/hand_wave.mp4"
output_path = "data/raw/hand_wave.json"


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import json

def extract_hand_landmarks(video_path, model_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open the video file {video_path}")
        exit()
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path= model_path),
    running_mode=VisionRunningMode.VIDEO)
    with HandLandmarker.create_from_options(options) as landmarker:
        ret, frame = cap.read()
        # print(f"{ret} frame read")
        # if ret == True:
        #     print(f"frame shape: {frame.shape}")
        while ret == True:
            ret, frame = cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    extract_hand_landmarks(video_path, model_path, output_path)