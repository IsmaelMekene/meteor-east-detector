
'''

What is the role of the datagenerator:

Generate data on the fly
with the notion of batch
when data can not fit in memory
Be careful you must check that u are using the right package i.e :

from tensorflow import keras

'''


import numpy as np
#import keras
from tensorflow import keras


#DataGenerator inherits from keras.utils.Sequence

#How is the "datagenerator" used ?
gen1 = DataGenerator(list_IDS, # may be a list or a dictionary of file ids
                     labels, # to each fileId corresponds a label
                     batch_size=8, # thse size of the batch ( will retrun batch_size *(image shape))
                     dim=(8*204*204), # dimension of the input of the network ~ tensor
                     n_channels=1, # n channels of input data (image)
                     n_classes=10, # number of classes in a classification pb (not our case : object detection pb)
                     shuffle=True # for each epoch, data is shuffled
                     )
                     
                     
 
class DataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  

  def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
               n_classes=10, shuffle=True):
      'Initialization'
      #  very Scikit-Learn-Like in the implementation
      # attribute names  similar to variable names

      self.dim = dim # tuple
      self.batch_size = batch_size # integer
      self.labels = labels # list
      self.list_IDs = list_IDs # list
      self.n_channels = n_channels # integer
      self.n_classes = n_classes # integer

      self.shuffle = shuffle # boolean

      self.on_epoch_end() # if we want the data to be shuffled (~indexes) we need to call this function at the initialization !

      #self.indexes = None ( we could have initialized an empty list of indexes)


  def __len__(self):
      'Denotes the number of batches per epoch'

      return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      # index = position we are interested in
      # for example index = 2
      # with batch_size = 4
      # indexes = self.indexes[2*4:(2+1)*4]
      # indexes =  self.indexes[8:12]  - which is compatible with a batch of size 4

      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
      # In this implementation : we have the list_IDs (original ids)
      # We have their index position (shuffled)
      # Now, we want to get the IDS corresponding to this shuffled data
      # comprehension list

      list_IDs_temp = [self.list_IDs[k] for k in indexes]
      # equivalent to :

      # list_IDs_temp = []
      # for k in indexes:
      #   list_IDs_temp.append(self.list_IDs[k])

      # Generate data
      X, y = self.__data_generation(list_IDs_temp)

      return X, y

  def on_epoch_end(self):
      'Updates indexes after each epoch'

      # function called @ each epoch end
      # list_IDS has been stored in self.list_IDs when initialized
      # we are taking all index

      # the attribute index ( self.index) did not exist at initialization
      self.indexes = np.arange(len(self.list_IDs))

      # we shuffle all index (@ each epoch)
      if self.shuffle == True: # boolean attribute
          np.random.shuffle(self.indexes)

  def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

      # Initialization
      # for using the input in NN
      # ll images should have the same size

      # create empty np.empty(batch_size, 32,32, 1)
      X = np.empty((self.batch_size, *self.dim, self.n_channels))

      # creates an empty array for y (same first dimension as X)
      y = np.empty((self.batch_size), dtype=int)

      # Generate data

      # thanks to enumerate we can get the index position and the value
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          # reads in the data stored as .npy arrays  those numpy arays are @ the right format !

          X[i,] = np.load('data/' + ID + '.npy')

          # Store class
          y[i] = self.labels[ID]

      # the output is transformed into 1-hot encoded vector
      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
      
      
      
      
      
#Notes

*Be careful:*
# we take the indices of each element in the array
# we rearrange the *indices*
# we will then use this 
my_array = np.arange(45)

#Be careful : np.random.shuffle directly modifies the data structure


#Why __len__ ?

To determine the number of batches produced by the datagenerator per epoch in our case it will be:
the (length of IDS) / batch size rounded to the lowest integer (np.floor).

#Why __get_item__? if we want to access : "batch[k]"

function(*(5,4,3)) = function(5,4,3)
