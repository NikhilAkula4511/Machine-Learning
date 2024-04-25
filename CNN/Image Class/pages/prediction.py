import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import os
st.write("Image_Classification")
modelss = load_model(r"C:\Users\Hello\Deep Learning\CNN\image_classification_without_augmantation.h5")

predictors = {0: 'bell pepper',
 1: 'espresso',
 2: 'koala bear',
 3: 'ladybug',
 4: 'lesser panda',
 5: 'lifeboat',
 6: 'orange',
 7: 'pizza',
 8: 'school bus',
 9: 'sports car'}

test_image = st.file_uploader("Testing Image",type=["JPEG"])
if test_image:
    img = os.path.join(f"C:\\Users\\Hello\\Deep Learning\\CNN\\data_1\\data\\class_10_val\\test_images\\{test_image.name}")
    image_array = cv2.imread(img)
    test_images = image_array[np.newaxis]
    a = modelss.predict(test_images)
    image_name = np.argmax(a)
    image_pred = predictors[image_name]
    col1,col2,col3 = st.columns(3)
    with col2:
        st.image(test_image, width=200) 
        st.write(f"<div style='text-align: center;'><span style='color:red;'>Prediction</span>&nbsp;&nbsp;<span style='color:green;'>{image_pred}</span></div>", unsafe_allow_html=True) 
    

