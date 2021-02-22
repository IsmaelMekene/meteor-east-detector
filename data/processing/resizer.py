import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob2

import PIL
try:
    import Image
except ImportError:
    from PIL import Image
import cv2
from skimage import io, color

from tensorflow import keras
import tensorflow as tf 
tf.__version__

from keras.layers import *
from tqdm import tqdm 

import ast
import shutil





def resize_image(image_path):

  al = plt.imread(image_path)
  #print('the initial shape is:',al.shape)


  if len(al.shape) == 2:

    al = cv2.merge((al,al,al))

  
  elif al.shape[2] == 4:

    al = cv2.cvtColor(al, cv2.COLOR_BGRA2BGR)

  else:

    al = al


  w = al.shape[1]  #store the width
  h = al.shape[0]  #store the height
  c = al.shape[2]  #store the number of channels(3 in this case)





  #print(c)

  # In case image is horizontally orientated
  if w > h:
    combine = np.zeros((w,w,c))
    combino = combine
    for i in range(c):
      combino[int((w-h)/2):int(((w-h)/2)+h) ,: ,i] = al[:,:,i]
    resized = cv2.resize(combino, (480, 480), interpolation=cv2.INTER_NEAREST)

  # In case image is vertically orientated
  elif w < h:
    combine = np.zeros((h,h,c))
    combino = combine
    for i in range(c):
      combino[: ,int((h-w)/2):int(((h-w)/2)+w) ,i] = al[:,:,i]
    resized = cv2.resize(combino, (480, 480), interpolation=cv2.INTER_NEAREST)

  # In case image is squared
  else:
    resized = cv2.resize(al, (480, 480), interpolation=cv2.INTER_NEAREST)
  
  al = resized 

  #plt.imshow(al.astype(np.uint8))  # to Clip input data to the valid range for imshow with RGB data.
  #plt.show()
  #print('the final shape is:',al.shape)

  return al




def resize_mask(ali):
  
  w = ali.shape[1]  #store the width
  h = ali.shape[0]  #store the height
  #c = al.shape[2]  #store the number of channels(3 in this case)
  #print(c)
  al = ali.reshape((h,w))

  #print('the initial shape is:',al.shape)

  # In case image is horizontally orientated
  if w > h:
    combine = np.zeros((w,w))
    combine[int((w-h)/2):int(((w-h)/2)+h) ,:] = al[:,:]
    resized = cv2.resize(combine, (480, 480), interpolation=cv2.INTER_NEAREST)

  # In case image is vertically orientated
  elif w < h:
    combine = np.zeros((h,h))
    combine[: ,int((h-w)/2):int(((h-w)/2)+w)] = al[:,:]
    resized = cv2.resize(combine, (480, 480), interpolation=cv2.INTER_NEAREST)

  # In case image is squared
  else:
    resized = cv2.resize(al, (480, 480), interpolation=cv2.INTER_NEAREST)
  
  #print('the final shape of resized is:',resized.shape)
  alou = np.reshape(resized, (480, 480, 1)) 
 
  #plt.imshow(resized)  # to Clip input data to the valid range for imshow with RGB data.
  #plt.show()
  #print('the final shape is:',alai.shape)

  return alou



def resize_distancegeo(npy_path):

  al = npy_path
  #al = np.load(npy_path)
  
  #print('the initial shape is:',al.shape)

  w = al.shape[1]  #store the width
  h = al.shape[0]  #store the height
  c = al.shape[2]  #store the number of channels(3 in this case)
  #print(c)

  # In case image is horizontally orientated
  if w > h:
    combine = np.zeros((w,w,c))
    combino = combine
    for i in range(c):
      combino[int((w-h)/2):int(((w-h)/2)+h) ,: ,i] = al[:,:,i]
    resized = cv2.resize(combino, (480, 480), interpolation=cv2.INTER_NEAREST)

  # In case image is vertically orientated
  elif w < h:
    combine = np.zeros((h,h,c))
    combino = combine
    for i in range(c):
      combino[: ,int((h-w)/2):int(((h-w)/2)+w) ,i] = al[:,:,i]
    resized = cv2.resize(combino, (480, 480), interpolation=cv2.INTER_NEAREST)

  # In case image is squared
  else:
    resized = cv2.resize(al, (480, 480), interpolation=cv2.INTER_NEAREST)
  
  al = resized 

  #plt.imshow(al.astype(np.uint8))  # to Clip input data to the valid range for imshow with RGB data.
  #plt.show()
  #print('the final shape is:',al.shape)

  return al


