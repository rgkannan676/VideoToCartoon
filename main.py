import os
import torch
import cv2
from dexined_model.model import DexiNed
from tqdm import tqdm
import numpy as np
import random


# Import everything needed to
from moviepy.editor import *

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

#Set maximum shape to image input to model, this is needed to avoid performance issues and CUDA memory
MODEL_INPUT_IMAGE_MAX_SIZE = 720

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

#Print config details.
def print_start():
    print("Starting Video To Cartoon Converter tool. Check https://github.com/rgkannan676/VideoToCartoon for more details.")
    print("See Configuration info are below. If required can change in main.py.")
    print("#Set White background and black edge : ", WHITE_BACKGROUND_BLACK_EDGE)
    print("#Edge threshold value for sharper edges : ", EDGE_THRESHOLD_ADJUST)
    print("#Add mix frame effect torandomly change image colours : ", ADD_MIX_COLOR_EFFECT)
    print("#Maximum shape of input image to model set to : ", MODEL_INPUT_IMAGE_MAX_SIZE)


#Using MoviePy librarty (https://zulko.github.io/moviepy/getting_started/getting_started.html#getting-started-with-moviepy) to set audio to edited video.
def add_audio_to_video(original_video_path, edited_video_path):
    original_video = VideoFileClip(original_video_path)
    original_audio = original_video.audio
    edited_video = VideoFileClip(edited_video_path)
    edited_video.set_audio(original_audio)
    edited_video.write_videofile(edited_video_path, verbose=False, progress_bar=False)



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

    #Control input size.
    if image.shape[0] > MODEL_INPUT_IMAGE_MAX_SIZE or image.shape[1] > MODEL_INPUT_IMAGE_MAX_SIZE:
        new_width = None
        new_height = None
        if image.shape[0] >=  image.shape[1]:
            new_height =  MODEL_INPUT_IMAGE_MAX_SIZE
            new_width = int(MODEL_INPUT_IMAGE_MAX_SIZE * (image.shape[1]/image.shape[0]))
        else:
            new_height = int(MODEL_INPUT_IMAGE_MAX_SIZE * (image.shape[0] / image.shape[1]))
            new_width = MODEL_INPUT_IMAGE_MAX_SIZE
        image = cv2.resize(image, (new_width, new_height))

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

    print_start()

    if torch.cuda.device_count() == 0:
        device = 'cpu'
    else:
        device = 'cuda'

    print("#Running on device : ", device)
    model = get_dexined_model(device)

    video_files_list = [x for x in os.listdir(VIDEO_INPUT_FOLDER) if x.endswith(".mkv") or x.endswith(".avi") or x.endswith(".mp4") or x.endswith(".webm")]
    print("#Number of videos found in folder ",VIDEO_INPUT_FOLDER, " : ", len(video_files_list))
    print("-------------------------------------------------------------------------------------")
    for vid_num, video in enumerate(video_files_list):
        print("Processing video number : " , str(vid_num + 1), " with name : ", video)
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
            print("ERROR : Error opening video stream or file")

        complete_percentage = 0

        # Read until video is completed
        while (cap.isOpened()):
            for complete_percentage in tqdm(range(num_frames),desc="Processing video frames : "):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret == True:
                    #For better perfomance and avoid memory issues, image is resized to half the size.
                    preprocessed_image = torch.unsqueeze(preprocess_image(frame), 0).to(device)
                    preds = model(preprocessed_image)
                    result_image = postprocess_image(preds,width,height)
                    output_file.write(result_image)
                    complete_percentage = complete_percentage + 1
                else:
                    break
            break


        # When everything done, release the video capture object
        cap.release()
        output_file.release()

        add_audio_to_video(video_path, output_path)

        print("Completed processing ", video, " Check folder ", VIDEO_OUTPUT_FOLDER, " for output video.")






    
