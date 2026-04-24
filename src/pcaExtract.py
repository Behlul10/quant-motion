from pathlib import Path
import sys
import json
import cv2
from sklearn.decomposition import PCA
import joblib
import numpy as np 

video_paths = Path(sys.argv[1]).glob("*.mp4") 
output_path= f"data/{Path(sys.argv[2])}"
# output_path = f"{Path(__file__).parent.parent}/data/pca/training_pca.json"

def extract_pca(video_paths, output_path):
    frame_list = []
    metadata_list = []
    final_result = []
 
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: could not open the video file {str(video_path)}")
            exit()
        ret, frame = cap.read() #read 1st frame
        while ret:
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gframe = cv2.resize(gframe, (256, 256))
            frame_list.append(gframe.flatten())
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            metadata_list.append({"video_path": str(video_path), "timestamp_ms": timestamp_ms})
            ret, frame = cap.read() #keep reading frame while frame return true
    
    if Path("models/quant/pca_model.pkl").exists():
        pca = joblib.load("models/quant/pca_model.pkl")
        final_pca_data = pca.transform(np.array(frame_list)).tolist()
    else:
        pca = PCA(n_components = 10)
        final_pca_data = pca.fit_transform(np.array(frame_list)).tolist()
        joblib.dump(pca, "models/quant/pca_model.pkl")
   
    for i in range(len(metadata_list)):
        final_result.append({
            "video_path": metadata_list[i]["video_path"],
            "timestamp_ms": metadata_list[i]["timestamp_ms"],
            "PCA_Data": final_pca_data[i]
        })


    with open(output_path, "w") as f:
        json.dump(final_result, f, indent=2)

if __name__ == "__main__":
    extract_pca(video_paths, output_path)