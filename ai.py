import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from skimage.filters import threshold_otsu
import numpy as np
from glob import glob
from scipy import misc 
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle
import os
from PIL import Image
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
#%matplotlib inline
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import RMSprop
#from keras.layers.normalization import BatchNormalization
data = glob('DB1_B/')
len(data)
images = []
def read_images(data):
    for i in range(len(data)):
        img = cv2.imread(data[i])
        img = cv2.resize(img,(224,224))
        images.append(img)
    return images
#images = read_images(data)
images_arr = np.asarray(images)
images_arr = images_arr.astype('float32')
images_arr.shape
# Shapes of training set
print("Dataset (images) shape: {shape}".format(shape=images_arr.shape))
#plt.figure(figsize=[5,5])
    
# Display the first image in training data
for i in range(2):
   plt.figure(figsize=[5, 5])
   curr_img = np.reshape(images_arr[i], (224,224))
   plt.imshow(curr_img, cmap='gray')
   plt.show()
images_arr = images_arr.reshape(-1, 224,224, 1)
images_arr.shape
images_arr.dtype
np.max(images_arr)
images_arr = images_arr / np.max(images_arr)





