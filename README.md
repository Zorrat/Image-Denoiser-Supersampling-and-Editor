# Image-Denoiser-Supersampling-and-Editor
Image Upsampling,Denoising and editor app created for Codestorm hackathon.
This project utilizes 2 AI based image upsampling models to upsample ur images by 3x scale. All editing functions are implemented with OpenCV 4.3.0

The models used in this project are
https://arxiv.org/abs/1707.02921

https://arxiv.org/abs/1608.00367






Dependencies

conda install -c conda-forge opencv=4.3.0

pip install streamlit


To run the project rename Directory to "ImageUpscaleProject" and Place it in C:/Users/'UserName"/
on terminal run

streamlit run ImageUpscaleProject/main.py
