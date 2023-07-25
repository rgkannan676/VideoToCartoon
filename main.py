import os
import torch
import cv2
from dexined_model.model import DexiNed
import tqdm
import numpy as np

VIDEO_INPUT_FOLDER = "video_input"
VIDEO_OUTPUT_FOLDER = "video_output"
DEXINED_MODEL_FOLDER = "dexined_model"
DEXINED_CHECK_POINT = "10_model.pth"
PRE_PROCESS_MEAN_PIXEL_VALUE = [103.939,116.779,123.68]

def get_dexined_model(device):
    checkpoint_path = os.path.join(DEXINED_MODEL_FOLDER,DEXINED_CHECK_POINT)
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(checkpoint_path,map_location=device))
    model.eval()

    return model


def preprocess_image(image):
    if image.shape[0] % 16 != 0 or image.shape[1] % 16 != 0:
        img_width = ((image.shape[1] // 16) + 1) * 16
        img_height = ((image.shape[0] // 16) + 1) * 16
        image = cv2.resize(image, (img_width, img_height))

    image = np.array(image, dtype=np.float32)
    image -= PRE_PROCESS_MEAN_PIXEL_VALUE
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image.copy()).float()
    return image

def postprocess_image(preds):
    fuse_pred = preds[6]
    fuse_sigmoid = torch.sigmoid(fuse_pred).cpu().detach().numpy()

    pass


if __name__ == '__main__':

    if torch.cuda.device_count() == 0:
        device = 'cpu'
    else:
        device = 'cuda'

    print("Running on device : ", device)
    model = get_dexined_model(device)

    video_files_list = [x for x in os.listdir(VIDEO_INPUT_FOLDER) if x.endswith(".mkv") or x.endswith(".avi") or x.endswith(".mp4") or x.endswith(".webm")]
    for video in video_files_list:
        print("Processing video : ", video)
        video_path = os.path.join(VIDEO_INPUT_FOLDER,video)

        cap = cv2.VideoCapture(video_path)
        #Get video parameters.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = os.path.join(VIDEO_OUTPUT_FOLDER,video)
        output_file = cv2.VideoWriter(
            filename=output_path,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            # cv2.VideoWriter_fourcc(*codec),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )


        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        complete_percentage = 0

        # Read until video is completed
        while (cap.isOpened()):
            for complete_percentage in tqdm.tqdm(range(num_frames)):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                    preprocessed_image = torch.unsqueeze(preprocess_image(frame), 0).to(device)
                    preds = model(preprocessed_image)
                    print("Len pf preds : ", len(preds))

                    output_file.write(frame)
                    complete_percentage = complete_percentage + 1
                else:
                    break
            print("Completed processing ", video, " Check folder ", VIDEO_OUTPUT_FOLDER, " for output video.")
            break


        # When everything done, release the video capture object
        cap.release()
        output_file.release()




    
