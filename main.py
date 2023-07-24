import os
import torch
import cv2
from dexined_model.model import DexiNed

VIDEO_INPUT_FOLDER = "video_input"
VIDEO_OUTPUT_FOLDER = "video_output"

def get_dexined model:
    


if __name__ == '__main__':
    video_files_list = [x for x in os.listdir(VIDEO_INPUT_FOLDER) if x.endswith(".mkv") or x.endswith(".avi") or x.endswith(".mp4") or x.endswith(".webm")]
    for video in video_files_list:
        print("Processing video : ", video)
        video_path = os.path.join(VIDEO_INPUT_FOLDER,video)
        cap = cv2.VideoCapture(video_path)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                ##IMAGES


        # When everything done, release the video capture object
        cap.release()




    
