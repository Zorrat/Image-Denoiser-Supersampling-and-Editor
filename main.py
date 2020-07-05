
####### CORE IMPORTS ###############
import cv2

from cv2 import dnn_superres
import streamlit as st
from PIL import Image,ImageEnhance
import time
import numpy as np
import os
import random
import string
from datetime import datetime

#Helper Functions

@st.cache(suppress_st_warning=True)
def loadim(img_file):
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img

def upScaleEDSR(image):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "ImageUpscaleProject/weights/EDSR_x3.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 3)
    result = sr.upsample(image)
    return result
    #cv2.imwrite('SuperResTest/UpscaledIm/'+ saveName, result)

def crop(img,x,y):
    y1,x1 = x
    y2,x2 = y
    crop_img = img[y1:y2,x1:x2]
    return crop_img

def upScaleFSRCNN(image):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    path = "ImageUpscaleProject/weights/FSRCNN_x3.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("fsrcnn", 3)
    result = sr.upsample(image)
    return result
   
    
    
def resize(img,shape):

    img_resized = cv2.resize(img, shape,interpolation = cv2.INTER_CUBIC) 
    return img_resized


    
def BilinearUpscaling(img,factor = 2):
    
    img_resized = cv2.resize(img,(0,0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    return img_resized
    
def rotate(image,x):
    (h1, w1) = image.shape[:2]
    center = (w1 / 2, h1 / 2)
    Matrix = cv2.getRotationMatrix2D(center, -90 * x, 1.0)
    rotated_image = cv2.warpAffine(image, Matrix, (w1, h1))
    return rotated_image


def denoise(img):

    # denoising of image saving it into dst image 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    return dst
    
def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def save(typ,img):
    now = datetime.now()
    time_stamp = now.strftime("%m_%d_%H_%M_%S") 
    fn ='ImageUpscaleProject/Saved_Images/'+typ+time_stamp+'.png'
    cv2.imwrite(fn,img)
    



def main():
    st.title("Image Super Sampling and Denoiser")
    st.subheader("App for AI based Image super resolution for image Upscaling,Denoising and Editing")
    
    #Uploading Main Image
    img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    if img_file is None:
        st.info("Please Select an Image")
    else:
        
        #Convert Selected image to opencv format
        
        main_img_cv = loadim(img_file)
        st.image(main_img_cv,channels = 'BGR',use_column_width = True)

        # Sidebar activites defined
        st.sidebar.subheader("Select which Editing Function you would like to Use")
        activities = ['Super Sampling','Denoise','Resize','Filter','Rotate','Crop','About']
        sidebar_choice = st.sidebar.selectbox('Select a feature',activities)
        #Sidebar activites
        if sidebar_choice == 'Super Sampling':
            st.subheader("Image Super resolution")
            upscale_type = st.selectbox("Select Upsampling Method",['EDSR','FSRCNN','Bilinear'])
            if upscale_type == 'EDSR':
                st.info('The Enhanced deep residual super sampling method is based on a larger model and will take anywhere from 2 - 15 min to complete upsampling.')
                if st.button("Apply and Save"):
                    if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
                        st.error('Image to Large for upsampling. This image is already above 1080 pixels wide')
                    else:
                        with st.spinner("Upscaling. Please Wait. This may take long."):
                            upscaled_image_cv = upScaleEDSR(main_img_cv)
                        
                        st.image(upscaled_image_cv, channels="BGR")
                        st.success("Image has been Upscaled")
                        save('EDSR_',upscaled_image_cv)
                        st.success("File has been Saved")
                        st.balloons()

            elif upscale_type == 'FSRCNN':
                st.info('FSRCNN is small fast model of quickly upscaling images with AI. This model does not produce state of the art accuracy however')
                
                if st.button("Apply"):
                    if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
                        st.error('Image to Large for upsampling. This image is already above 1080 pixels wide')
                    else:
                        with st.spinner("Upscaling... Hold on Tight this may take a cuple of seconds"):
                            upscaled_image_cv = upScaleFSRCNN(main_img_cv)
                        st.image(upscaled_image_cv, channels="BGR",format = 'png')
                        st.success("Image has been Upscaled")
                        #save('FSRCNN_',upscaled_image_cv)
                        #st.success("File has been Saved")
                if st.button("Apply and Save"):
                    if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
                        st.error('Image to Large for upsampling. This image is already above 1080 pixels wide')
                    else:
                        with st.spinner("Upscaling... Hold on Tight this may take a cuple of seconds"):
                            upscaled_image_cv = upScaleFSRCNN(main_img_cv)
                        st.image(upscaled_image_cv, channels="BGR",format = 'png')
                        st.success("Image has been Upscaled")
                        save('FSRCNN_',upscaled_image_cv)
                        st.success("File has been Saved")    
            elif upscale_type == 'Bilinear':
                st.info('Bilinear is a non AI Image upscaling algorithm based on Bilinear interpolation.')
                bilinear_factor = st.slider('Select Upscale Factor',2,4)
                if st.button("Apply"):
                    upscaled_image_cv = BilinearUpscaling(main_img_cv,bilinear_factor)
                    st.image(upscaled_image_cv, channels="BGR",format = 'png')
                    st.success("Image has been Upscaled with Bilinear Interpolation")

                if st.button("Apply and Save"):
                    upscaled_image_cv = BilinearUpscaling(main_img_cv,bilinear_factor)
                    st.image(upscaled_image_cv, channels="BGR",format = 'png')
                    st.success("Image has been Upscaled with Bilinear Interpolation")
                    save('bilinear',upscaled_image_cv)
                    st.success("File has been Saved")
        if sidebar_choice == 'Denoise':
            st.subheader("Image Denoiser")
            st.info('The Image denoiser will remove noise from Images. Implementation via OpenCV')
            if st.button("Apply"):
                with st.spinner("Denoising.. Please Hold On to your seat belts"):
                    denoise_image_cv = denoise(main_img_cv)
                    st.image(denoise_image_cv,channels="BGR",use_column_width = True)
                #save('Denoise_',denoise_image_cv)
                st.success("Image was Denoised")
                st.balloons()
            if st.button("Apply and Save"):
                with st.spinner("Denoising.. Please Hold On to your seat belts"):
                    denoise_image_cv = denoise(main_img_cv)
                    st.image(denoise_image_cv,channels="BGR",use_column_width = True)
                save('Denoise_',denoise_image_cv)
                st.success("Image was Denoised and saved")
                st.balloons()

        if sidebar_choice == 'Resize':
            st.subheader("Image Resize")
            st.markdown("Please Enter the Dimentions you would like to resize the image to.")
            dim = st.text_input("Enter Dimentions with a comma",'512,512')
            if st.button('Apply'):
                dim = dim.split(',')
                if len(dim)!=2:
                    st.error("Incorrect Dimentions")
                else:
                    shape = (int(dim[0]),int(dim[1]))
                    resize_image = resize(main_img_cv,shape)
                    st.image(resize_image,channels = 'BGR')
                    #save('Resize_',resize_image)
                    st.success("Image was Resized")
            if st.button('Apply and save'):
                dim = dim.split(',')
                if len(dim)!=2:
                    st.error("Incorrect Dimentions")
                else:
                    shape = (int(dim[0]),int(dim[1]))
                    resize_image = resize(main_img_cv,shape)
                    st.image(resize_image,channels = 'BGR')
                    save('Resize_',resize_image)
                    st.success("Image was Resized and saved")



        if sidebar_choice == 'Filter':
            st.subheader('Apply Various Filters and Effects to your Image.')
            filter = st.selectbox("Select a Filter to apply",['Color Sketch','Pencil Sketch','Negative'])
            if filter == 'Color Sketch':
                if st.button("Apply"):
                    color_sketch =  cv2.stylization(main_img_cv, sigma_s=60, sigma_r=0.07)
                    st.image(color_sketch,channels = 'BGR',use_column_width = True)
                    #save('C_Sketch_',color_sketch)
                    st.success("Filter is applied")
                if st.button("Apply and Save"):
                    color_sketch =  cv2.stylization(main_img_cv, sigma_s=60, sigma_r=0.07)
                    st.image(color_sketch,channels = 'BGR',use_column_width = True)
                    save('C_Sketch_',color_sketch)
                    st.success("Filter is applied and saved")


            if filter == 'Pencil Sketch':
                if st.button("Apply"):
                    pencil_sketch,_ = cv2.pencilSketch(main_img_cv, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
                    st.image(pencil_sketch,use_column_width = True)
                    #save('P_Sketch_',pencil_sketch)
                    st.success("Filter is applied and ")
                if st.button("Apply and Save"):
                    pencil_sketch,_ = cv2.pencilSketch(main_img_cv, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
                    st.image(pencil_sketch,use_column_width = True)
                    save('P_Sketch_',pencil_sketch)
                    st.success("Filter is applied and saved")

            if filter == 'Negative':
                if st.button("Apply"):
                    negative_image = cv2.bitwise_not(main_img_cv)
                    st.image(negative_image,channels = 'BGR',use_column_width = True)
                    #save('Negative',negative_image)
                    st.success("Filter is applied.")
                if st.button("Apply and save"):
                    negative_image = cv2.bitwise_not(main_img_cv)
                    st.image(negative_image,channels = 'BGR',use_column_width = True)
                    save('Negative',negative_image)
                    st.success("Filter is applied and saved")
        
        if sidebar_choice == 'Crop':
            st.subheader("Image Cropper")
            st.info(' Enter The location of upper left and bottom right pixels to crop the image ')
            st.write('Your Image Dimentions are',main_img_cv.shape)
            x = st.text_input('Enter Upper left pixel location','100,100')
            y = st.text_input('Enter Bottom Right Pixel Location','300,300')
            if st.button('Apply'):
                x = x.split(",")

                x = (int(x[0]),int(x[1]))
                y = y.split(',')

                y = (int(y[0]),int(y[1]))
                st.write('Cropping to Dimetions',x,y)
                crop_image = crop(main_img_cv,x,y)
                st.image(crop_image,channels="BGR",use_column_width = True)
                #save('crop_',crop_image)
                st.success("Image Cropped.")
            if st.button('Apply and Save'):
                x = x.split(",")

                x = (int(x[0]),int(x[1]))
                y = y.split(',')

                y = (int(y[0]),int(y[1]))
                st.write('Cropping to Dimetions',x,y)
                crop_image = crop(main_img_cv,x,y)
                st.image(crop_image,channels="BGR",use_column_width = True)
                save('crop_',crop_image)
                st.success("Image Cropped and saved.")
        
        if sidebar_choice =='Rotate':
            st.subheader("Image Rotating")
            st.info("Enter the ammount of times u want to rotate image to the Right. Select 3 for one left rotation")
            r_times = st.slider('Select number of times to rotate image right',1,3)
            if st.button('Apply'):
                rotated_image = rotate(main_img_cv,r_times)
                st.image(rotated_image,channels="BGR",use_column_width = True)
                #save('rotated_',rotated_image)
                st.success("Image Rotated.")
            if st.button('Apply and save'):
                rotated_image = rotate(main_img_cv,r_times)
                st.image(rotated_image,channels="BGR",use_column_width = True)
                save('rotated_',rotated_image)
                st.success("Image Rotated and saved.")
        
        if sidebar_choice == 'About':
            st.header("Welcome to Image super resolution, Denoiser and Image Editor app.")
            st.subheader(" We have a deployed to AI based Image super resolution models in this app to upscale images")
            st.info("All Image processing is done via OpenCV")
            st.info('Front End Deployed on Streamlit.')
            st.info("This Program was created for the Codestorm Hackathon 2020")
            st.subheader("These Image upscaling Models are based on the Following papers")
            st.write('https://arxiv.org/abs/1707.02921')
            st.write('https://arxiv.org/abs/1608.00367')



if __name__ == '__main__':
    main()
    


