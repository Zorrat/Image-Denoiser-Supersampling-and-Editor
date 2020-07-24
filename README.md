# AI Image SuperSamping,Denoiser and Image Editor App.

This app was created for the Codestorm hackathon.
This project utilizes 2 AI based image upsampling models to upsample  images by a scale of 3x.
All editing functions are implemented with OpenCV 4.3.0.
The models used in this project are

https://arxiv.org/abs/1707.02921

https://arxiv.org/abs/1608.00367

Requirements :
Create a virtual conda enviornment since opencv = 4.3 may clash with spyder.

    conda create --name vEnv1 python=3.7
    conda activate vEnv1
    conda install -c conda-forge opencv=4.3.0
    pip install streamlit

Usage

    streamlit run main.py


Demo Video of the working.
Click the magic butterfly :)

[![Watch the video](https://github.com/Zorrat/Image-Denoiser-Supersampling-and-Editor/blob/master/LowRes/butterfly.png)](https://dms.licdn.com/playlist/C5605AQF5pyMCug-E-g/mp4-720p-30fp-crf28/0?e=1595689200&v=beta&t=fROqjVfYpQ7wMv52xeLWle6Eodkn0R4vtLEOBqd-ycI)
