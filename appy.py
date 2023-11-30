#!/usr/bin/env python
# coding: utf-8

# In[7]:

import plotly.express as px
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


# In[12]:


import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[16]:


def execute_model(image_file):
    new_model = load_model(os.path.join('models','malaria.h5'))
    #st.write(type(image_file))
    image_2=cv2.imread(image_file.name)
    fig = px.imshow(image_2)
    resize_2 = tf.image.resize(image_2,(256,256))
    #fig = px.imshow(resize_2.numpy().astype(int))
    #fig=plt.imshow(resize_2.numpy().astype(int))
    st.plotly_chart(fig, use_container_width=True)
    yhat = new_model.predict(np.expand_dims(resize_2/255,0))
    if yhat[0][0] >0.50:
        st.success('The person is Uninfected')
    else:
        st.success('The person is Parasitized')    


# In[15]:



def main():       
    # front end elements of the web page 
    st.title("Malaria Detection")
    image_file = st.file_uploader("Upload Image File",type=['jpg','bmp','png','jpeg'])      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        execute_model(image_file)
     
 
if __name__=='__main__': 
    main()


# In[ ]:




