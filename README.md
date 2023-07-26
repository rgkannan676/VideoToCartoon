# VideoToCartoon

A tool that converts video into black-and-white cartoons. This repo uses the edge detection model provided by **[DexiNed](https://github.com/xavysp/DexiNed)**. 


# Installation and Processing Steps

Steps to install and use in Ananconda
- conda create --name videoToCartoon python=3.8
- conda activate videoToCartoon
- git clone https://github.com/rgkannan676/VideoToCartoon.git
- cd VideoToCartoon
- Install latest PyTorch from 'https://pytorch.org/' example: 'conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia'
- Install the required libraries: pip install -r requirements.txt
- Download dexined pytorch checkpoint model 10_model.pth provided by **[DexiNed](https://github.com/xavysp/DexiNed)**  and copy to 'dexined_model' folder. 
- Copy the videos to covert in the folder 'video_input'
- Run 'python main.py'. This will start the video edit processing.
- See the output videos in folder 'video_output'
