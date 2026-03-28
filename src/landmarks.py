import sys
import mediapipe as mp
import cv2
import json
from pathlib import Path


model_path = f"{Path(__file__).parent.parent}/models/hand_landmarker.task"
video_path = sys.argv[1]
output_path = f"{Path(__file__).parent.parent}/data/raw/{Path(video_path).stem}.json"

def extract_hand_landmarks(video_path, model_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open the video file {video_path}")
        exit()
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    landmarks = []

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path= model_path),
    running_mode=VisionRunningMode.VIDEO)
    with HandLandmarker.create_from_options(options) as landmarker:
        ret, frame = cap.read()
        while ret == True:
            ret, frame = cap.read()
            if ret == False:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.hand_landmarks: #if list is not empty
                temp_list = []
                for point in result.hand_landmarks[0]:
                    temp_list.append([point.x, point.y, point.z])
                landmarks.append({"timestamp": timestamp_ms, "landmarks": temp_list})
    with open(output_path, "w") as f:
        json.dump(landmarks, f, indent=2)

if __name__ == "__main__":
    extract_hand_landmarks(video_path, model_path, output_path)