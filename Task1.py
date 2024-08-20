#!/usr/bin/env python
# coding: utf-8

# 1. Visualize Activation Maps: Description: Visualize activation maps to understand which image regions activate CNN filters for emotion detection. Guidelines: You can use any of your pre trained model (made by you) for this task. GUI is not necessary for this task.

# In[1]:


get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install --upgrade keras tensorflow')
get_ipython().system('pip install --upgrade opencv-python')


# In[3]:


pip install --upgrade pip


# In[6]:


#importing the dependencies


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[13]:


# Loading the pre-trained model
model = VGG16(weights='imagenet', include_top=False)


# In[14]:


# preprocessesing of the model
img_path = 'beautiful_girl.jpg'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# In[15]:


# Get activations for a specific layer
layer_name = 'block5_conv3'  # Replace with desired layer
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(x)


# In[11]:


# Generate heatmap for visualization
heatmap = np.mean(intermediate_output[0], axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()


# In[1]:





# In[ ]:




