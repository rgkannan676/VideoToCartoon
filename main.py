import os
import torch
import cv2
from dexined_model.model import DexiNed
import tqdm
import numpy as np
import random

######## No Change Needed Configs ###########
#Input Video Folder
VIDEO_INPUT_FOLDER = "video_input"

#Output Video Folder
VIDEO_OUTPUT_FOLDER = "video_output"

#Dexined Model model
DEXINED_MODEL_FOLDER = "dexined_model"

#Dexined checkpoint name provided by https://github.com/xavysp/DexiNed/tree/master
DEXINED_CHECK_POINT = "10_model.pth"

#Mean pixel values provided in the repo https://github.com/xavysp/DexiNed/tree/master
PRE_PROCESS_MEAN_PIXEL_VALUE = [103.939,116.779,123.68]

###############################################

######## Changable Configs ####################
#For large images, takes more time and memmory.
MODEL_INPUT_IMAGE_SCALE = 0.5

# This config controls background and edge colour.
#If True :Background white, edge balck
#If False :Background black, edge white
WHITE_BACKGROUND_BLACK_EDGE = True

# If the value is higher, pixels with less confidence will be set as background pixel giving more sharp video.
EDGE_THRESHOLD_ADJUST = 0

# This is jus to add some random effects mixing the background and edge colours
# If True : Add effect randomly.
#If False : Follow WHITE_BACKGROUND_BLACK_EDGE variable.
ADD_MIX_COLOR_EFFECT = False

###############################################


#Initialize dexined model with checkpoint weights.
def get_dexined_model(device):
    checkpoint_path = os.path.join(DEXINED_MODEL_FOLDER,DEXINED_CHECK_POINT)
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(checkpoint_path,map_location=device))
    model.eval()

    return model

#Taken from https://github.com/xavysp/DexiNed/blob/master/utils/image.py
def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


#Preprocess image
# 1) resize image to multiple of 16
# 2) subtract mean and transpose according to preprocess steps in https://github.com/xavysp/DexiNed/blob/master/datasets.py
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

#Post process prediction results
#Follow steps in https://github.com/xavysp/DexiNed/blob/master/utils/image.py
#Here, considering only fused layer(6)
def postprocess_image(preds,width,height):
    fuse_pred = preds[6]
    fuse_sigmoid = torch.sigmoid(fuse_pred).cpu().detach().numpy()
    tmp_img = cv2.bitwise_not(np.uint8(image_normalization(fuse_sigmoid))).astype(np.uint8)[0,0,:,:]

    if EDGE_THRESHOLD_ADJUST > 0:
        tmp_img[tmp_img > EDGE_THRESHOLD_ADJUST] = 255

    if (not WHITE_BACKGROUND_BLACK_EDGE) and (not ADD_MIX_COLOR_EFFECT):
        tmp_img = cv2.bitwise_not(tmp_img)

    if ADD_MIX_COLOR_EFFECT:
        if random.random() > 0.5:
            tmp_img = cv2.bitwise_not(tmp_img)


    tmp_img = cv2.merge((tmp_img, tmp_img, tmp_img))
    tmp_img = cv2.resize(tmp_img, (width, height))
    return tmp_img


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
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), #Can depend on environment this runs. If fails check what is compactible in your environment
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
                    #For better perfomance and avoid memory issues, image is resized to half the size.
                    frame = cv2.resize(frame, (int(width * MODEL_INPUT_IMAGE_SCALE ), int(height * MODEL_INPUT_IMAGE_SCALE)))
                    preprocessed_image = torch.unsqueeze(preprocess_image(frame), 0).to(device)
                    preds = model(preprocessed_image)
                    result_image = postprocess_image(preds,width,height)
                    output_file.write(result_image)
                    complete_percentage = complete_percentage + 1
                else:
                    break
            print("Completed processing ", video, " Check folder ", VIDEO_OUTPUT_FOLDER, " for output video.")
            break


        # When everything done, release the video capture object
        cap.release()
        output_file.release()




    
