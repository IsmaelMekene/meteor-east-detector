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





#myframe = pd.read_csv('the 3 columns dataframe containing the names of each files (1: images, 2: masks, 3: distancedeo)')




class DataGeneratorPspnet(keras.utils.Sequence):
    'Generates data for Keras'


        #'Initialization'
    
    def __init__(self, batch_size, dataframe, input_size = 480, shuffle=True):


      self.batch_size = batch_size
      self.dataframe = dataframe
      self.shuffle = shuffle  #NOTE that the SHUFFLE is at the beginning of each epoch!!!
      self.input_size = input_size
      self.on_epoch_end()

      unik = self.dataframe['imageName'].unique().tolist()
      sort_unik = sorted(unik)
      frame_images = pd.DataFrame(sort_unik, columns=['names']).sort_values(by=['names'], ascending=True).reset_index(drop=True)
  


    def __len__(self):
        'Denotes the number of batches per epoch'

        unik = self.dataframe['imageName'].unique().tolist()
        sort_unik = sorted(unik)
        frame_images = pd.DataFrame(sort_unik, columns=['names']).sort_values(by=['names'], ascending=True).reset_index(drop=True)

        return int(np.floor(len(frame_images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        unik = self.dataframe['imageName'].unique().tolist()
        sort_unik = sorted(unik)
        frame_images = pd.DataFrame(sort_unik, columns=['names']).sort_values(by=['names'], ascending=True).reset_index(drop=True)

        # Find the kth corresponding batchsize dataframe
        df_temp = frame_images.iloc[index*self.batch_size:(index+1)*self.batch_size, :]

        # Generate data
        larray = self.name_generation(df_temp, self.batch_size)
        
        return larray 


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        unik = self.dataframe['imageName'].unique().tolist()
        sort_unik = sorted(unik)
        frame_images = pd.DataFrame(sort_unik, columns=['names']).sort_values(by=['names'], ascending=True).reset_index(drop=True)

        self.indexes = np.arange(len(frame_images))
        if self.shuffle == True:    #if shuffle is activated
            np.random.shuffle(self.indexes)  #randomly generate the index


    def name_generation(self, batch_size):
      'Generates data names following batch_size samples'

      #k = 19
      n_batch = batch_size  #set number of batch

      #make a list of unique imageName values
      unik = self.dataframe['imageName'].unique().tolist()  
      
      sort_unik = sorted(unik)  #sort it

      #make a dataframe from unique  
      frame_images = pd.DataFrame(sort_unik, columns=['names']).sort_values(by=['names'], ascending=True).reset_index(drop=True) 
      
      


      for k in range(int(len(frame_images)/batch_size)):#iterate over the batches 
      
        #dataframe for each batch
        boom = frame_images.iloc[k*n_batch:(k+1)*n_batch, :] 

        #empty array of size (batch_size,input,input,3)
        X = np.empty((batch_size, self.input_size, self.input_size, 3))

        for j, pure in enumerate (boom['names']):

        #resizing the images

          al = resize_image(f'/content/raw_Images/raw_Images/{pure}.jpg')

          #pure_images.append(al)
          X[j] = al



        #empty array of size (batch_size,input,input,1)
        Y_1 = np.empty((batch_size, int(self.input_size), int(self.input_size), 1))
        
        for zk, noms in enumerate (boom['names']):

          #make a dataframe with the imageNames in boom
          groupe_of_first_image = self.dataframe[self.dataframe['imageName'] == noms]
          groupe = groupe_of_first_image.reset_index(drop=True) #drop index
          

          #column 7 is for image_height
          #column 6 is for image_width
          #column 5 is for Y2
          #column 4 is for X2
          #column 3 is for Y1
          #column 2 is for X1
          #shape of matrix is (image_height, image_width)
          matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          for i in range(len(groupe)):
              #shape of block is (image_height, image_width)
              block = (np.zeros(((groupe.iloc[i, 7]), (groupe.iloc[i, 6]))))

              #shape of matrix is (image_height, image_width)
              block[(groupe.iloc[i, 3]):(groupe.iloc[i, 5]), (groupe.iloc[i, 2]):(groupe.iloc[i, 4])] = 1

              matrix = np.add(matrix, block) #matrix + block
              matrix[matrix>1]=1

          #print('the shape of matrix is:', matrix.shape)


          ali = resize_mask(matrix)
          #ali = matrix 

          Y_1[zk] = ali


        

        Y_2 = np.empty((batch_size, int(self.input_size), int(self.input_size), 4))
        for zi, nombre in enumerate (boom['names']):
          
          groupe_of_first_image = self.dataframe[self.dataframe['imageName'] == nombre]  #enumerate only the first image
          groupe = groupe_of_first_image.reset_index(drop=True)  #drop index


          #gloc = (np.zeros((4, (groupe.iloc[0, 7]), (groupe.iloc[0, 6])))) #the big tensor of dim(4,m,n)
          flop = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))  #large matrix of dim(m,n)
          
          x_up_left_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          y_up_left_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          
          x_up_right_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          y_up_right_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          
          x_down_right_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          y_down_right_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          
          x_down_left_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          y_down_left_matrix = (np.zeros(((groupe.iloc[0, 7]), (groupe.iloc[0, 6]))))
          
          
          
          width = (groupe.iloc[0, 6])  #store the width
          height = (groupe.iloc[0, 7])  #store the height
          x_indices = np.indices((height, width))[1] #matrix of x indices
          y_indices = np.indices((height, width))[0] #matrix of y indices
          #print(x_indices)
          #print(y_indices)
          
          #break

          for i in range(len(groupe)):  #iterate over the length of the 'group'



              block = (np.zeros(((groupe.iloc[i, 7]), (groupe.iloc[i, 6]))))   #large matrix of dim(m,n)



              block[(groupe.iloc[i, 3]):(groupe.iloc[i, 5]), (groupe.iloc[i, 2]):(groupe.iloc[i, 4])] = 1  #fill the block w/ 1s

              flop = np.add(flop, block) #matrix + block

              x_up_left = (groupe.iloc[i, 2])  #x of upleft corner of bbox
              y_up_left = (groupe.iloc[i, 3])  #y of upleft corner of bbox

              x_up_left_matrix += block*x_up_left
              y_up_left_matrix += block*y_up_left
              

              x_up_right = (groupe.iloc[i, 4])  #x of upright corner of bbox
              y_up_right = (groupe.iloc[i, 3])  #y of upright corner of bbox
              
              x_up_right_matrix += block*x_up_right
              y_up_right_matrix += block*y_up_right


              x_down_right = (groupe.iloc[i, 4])  #x of downright corner of bbox 
              y_down_right = (groupe.iloc[i, 5])  #y of downright corner of bbox 
              
              x_down_right_matrix += block*x_down_right
              y_down_right_matrix += block*y_down_right
              

              x_down_left = (groupe.iloc[i, 2])  #x of downleft corner of bbox 
              y_down_left = (groupe.iloc[i, 5])  #y of downleft corner of bbox 
              
              x_down_left_matrix += block*x_down_left
              y_down_left_matrix += block*y_down_left

              
              
          x_indices_mask = np.multiply(flop, x_indices)  #matrix with masks for x indices 
          y_indices_mask = np.multiply(flop, y_indices)  #matrix with masks for y indices 



          distance_from_up_left = (x_indices_mask - x_up_left_matrix)**2 + (y_indices_mask - y_up_left_matrix)**2  #compute the distance
          distance_from_up_left = np.multiply(distance_from_up_left,flop) #hadamard product with the masks
          distance_from_up_left = np.abs(distance_from_up_left) #absolute values 
          distance_from_up_left = np.sqrt(distance_from_up_left) #square root
          
          distance_from_up_right = (x_indices_mask - x_up_right_matrix)**2 + (y_indices_mask - y_up_right_matrix)**2  #compute the distance
          distance_from_up_right = np.multiply(distance_from_up_right,flop) #hadamard product with the masks
          distance_from_up_right = np.abs(distance_from_up_right) #absolute values 
          distance_from_up_right = np.sqrt(distance_from_up_right) #square root

          distance_from_down_right = (x_indices_mask - x_down_right_matrix)**2 + (y_indices_mask - y_down_right_matrix)**2 #compute the distance
          distance_from_down_right = np.multiply(distance_from_down_right,flop) #hadamard product with the masks
          distance_from_down_right = np.abs(distance_from_down_right) #absolute values 
          distance_from_down_right = np.sqrt(distance_from_down_right) #square root
          
          distance_from_down_left = (x_indices_mask - x_down_left_matrix)**2 + (y_indices_mask - y_down_left_matrix)**2 #compute the distance
          distance_from_down_left = np.multiply(distance_from_down_left,flop) #hadamard product with the masks
          distance_from_down_left = np.abs(distance_from_down_left) #absolute values 
          distance_from_down_left = np.sqrt(distance_from_down_left) #square root
          
          gloc = cv2.merge([distance_from_up_left, distance_from_up_right, distance_from_down_right, distance_from_down_left])
          
          ale = resize_distancegeo(gloc)
          
          #pure_distancegeo.append(ale)  # add to the list of distancegeo
          Y_2[zi] = ale

        



        list_de_sortie = []  #empty list
        list_de_sortie.append(X)  #add the list of images
        list_de_sortie.append(Y_1)   #add the list of the y
        #list_de_sortie.append(ali)
        
        #list_de_sortie.append(Y_2)        

        yield list_de_sortie







 
