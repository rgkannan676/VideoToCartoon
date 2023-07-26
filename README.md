# VideoToCartoon

A tool that converts video into black-and-white cartoons. This repo uses the edge detection model provided by **[DexiNed](https://github.com/xavysp/DexiNed)**. 


## Installation and Processing Steps

Steps to install and use in Ananconda
- conda create --name videoToCartoon python=3.8
- conda activate videoToCartoon
- git clone https://github.com/rgkannan676/VideoToCartoon.git
- cd VideoToCartoon
- Install the latest PyTorch from 'https://pytorch.org/' example: 'conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia'
- Install the required libraries: pip install -r requirements.txt
- Download dexined pytorch checkpoint model [10_model.pth](https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu/view?usp=sharing) provided by **[DexiNed](https://github.com/xavysp/DexiNed)**  and copy to 'dexined_model' folder. 
- Copy the videos to covert in the folder 'video_input'
- Run 'python main.py'. This will start the processing.
- See the output videos in folder 'video_output'

## Adjustable Configs
- MODEL_INPUT_IMAGE_MAX_SIZE: Set maximum shape to image input to the model. This is needed to avoid performance issues and CUDA memory.
- WHITE_BACKGROUND_BLACK_EDGE: This config controls background and edge colour. If set to True will make the background white and the edge black. If False, will set opposite.
- EDGE_THRESHOLD_ADJUST: If the value is higher, pixels with less confidence will be set as background pixels giving more sharp video.
- ADD_MIX_COLOR_EFFECT: This adds some random effects mixing the background and edge colours between white and black.  If set to True will Add effect randomly. If False, follow the WHITE_BACKGROUND_BLACK_EDGE setting.
- ADD_ORIGINAL_IMAGE_FRAMES: Add random frames of the original images effect.
- ORIGINAL_FRAMES_TO_ADD: Number of original frames to add randomly when ADD_ORIGINAL_IMAGE_FRAMES is True.
- ORIGINAL_FRAMES_TO_ADD_INTERVAL: Interval between frames before adding the next original set of frames when ADD_ORIGINAL_IMAGE_FRAMES is True.

## Result Sample
![Original](samples/original.PNG?raw=true "Original")
![Result](samples/result.PNG?raw=true "Result")
