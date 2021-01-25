




myframe = pd.read_csv('the 3 columns dataframe containing the names of each files (1: images, 2: masks, 3: distancedeo)')




class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    
        #'Initialization'
    
    def __init__(self, batch_size, df, shuffle=True):

      self.batch_size = batch_size
      self.df = df
      self.shuffle = shuffle



    def name_generation(self, dataframe, batch_size):
      'Generates data names following batch_size samples'

      #k = 19
      n_batch = batch_size
      for k in range(int(len(dataframe)/batch_size)):

        
        boom = dataframe.iloc[k*n_batch:(k+1)*n_batch, :]

        pure_images = []
        pure_masks = []
        pure_distancegeo = []

        X = np.empty((batch_size, 112, 112, 3))

        for j, pure in enumerate (boom['names']):

        #resizing the images

          al = plt.imread(pure)


          w = al.shape[1]  #store the width
          h = al.shape[0]  #store the height
          c = al.shape[2]  #store the number of channels(3 in this case)

          # In case image is horizontally orientated
          if w > h:
            up = np.zeros((int((w-h)/2),w))
            down = np.zeros((int((w-h)/2),w))

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the ideal padding and the resize
              caree = np.vstack((up,al[:,:,i],down))
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel) #merge into the same tensor



          # In case image is vertically orientated
          elif w < h:
            left = np.zeros((h,int((h-w)/2)))
            right = np.zeros((h,int((h-w)/2)))

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the ideal padding and the resize
              caree = np.vstack((left,al[:,:,i],right))
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel)  #merge into the same tensor


          # In case image is squared
          else:

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the resize
              caree = al[:,:,i]
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel)  #merge into the same tensor


          al = RESIZED 

          #pure_images.append(al)
          X[j] = al


        Y_1 = np.empty((batch_size, 112, 112, 3))
        for zk, mask in enumerate (boom['masks']):

          ali = plt.imread(mask)


          w = ali.shape[1]  #store the width
          h = ali.shape[0]  #store the height
          c = ali.shape[2]  #store the number of channels(3 in this case)

          # In case image is horizontally orientated
          if w > h:
            up = np.zeros((int((w-h)/2),w))
            down = np.zeros((int((w-h)/2),w))

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the ideal padding and the resize
              caree = np.vstack((up,ali[:,:,i],down))
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel) #merge into the same tensor



          # In case image is vertically orientated
          elif w < h:
            left = np.zeros((h,int((h-w)/2)))
            right = np.zeros((h,int((h-w)/2)))

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the ideal padding and the resize
              caree = np.vstack((left,ali[:,:,i],right))
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel) #merge into the same tensor


          # In case image is squared
          else:

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the resize
              caree = ali[:,:,i]
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel)  #merge into the same tensor


          ali = RESIZED 
          #pure_masks.append(ali)
          Y_1[zk] = ali



        Y_2 = np.empty((batch_size, 112, 112, 4))
        for zi, distance in enumerate (boom['distancegeo']):

          ale = np.load(distance)


          w = ale.shape[1]  #store the width
          h = ale.shape[0]  #store the height
          c = ale.shape[2]  #store the number of channels(3 in this case)

          # In case image is horizontally orientated
          if w > h:
            up = np.zeros((int((w-h)/2),w))
            down = np.zeros((int((w-h)/2),w))

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the ideal padding and the resize
              caree = np.vstack((up,ale[:,:,i],down))
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel)  #merge them into the same tensor



          # In case image is vertically orientated
          elif w < h:
            left = np.zeros((h,int((h-w)/2)))
            right = np.zeros((h,int((h-w)/2)))

            on_each_channel = []

            for i in range(c):
              #for each iteration, do the ideal padding and the resize
              caree = np.vstack((left,ale[:,:,i],right))
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel)  #mergen them into the same tensor


          # In case image is squared
          else:

            on_each_channel = [] 

            for i in range(c):
              #for each iteration, do the resize
              caree = ale[:,:,i]
              resized = cv2.resize(caree, (112, 112), interpolation=cv2.INTER_NEAREST)
              on_each_channel.append(resized)

            RESIZED = cv2.merge(on_each_channel)  #merge them into the same tensor


          ale = RESIZED 
          #pure_distancegeo.append(ale)  # add to the list of distancegeo
          Y_2[zi] = ale

        


        #les_y = []  #empty list
        #les_y.append(pure_masks)  #add the list of masks 
        #les_y.append(pure_distancegeo)  # add the list of distancegeo

        list_de_sortie = []  #empty list
        list_de_sortie.append(X)  #add the list of images
        list_de_sortie.append(Y_1)   #add the list of the y
        list_de_sortie.append(Y_2)        

        yield list_de_sortie   #return this list of two lists



 


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

            X[i,] = np.load('data/' + ID + '.npy')  #load the image (np.array)

            w = X[i,].shape[1]  #store the width
            h = X[i,].shape[0]  #store the height
            
            # In case image is horizontally orientated
            if w > h:
              up = np.zeros(((w-h)/2,w))
              down = np.zeros(((w-h)/2,w))
              carree = np.vstack((up,X[i,],down))  #do a vertical padding
              resized = cv2.resize(carree, (112, 112), interpolation=cv2.INTER_NEAREST) #resize 

            # In case image is vertically orientated
            elif w < h:
              left = np.zeros((h,(h-w)/2))
              right = np.zeros((h,(h-w)/2))
              square = np.column_stack((left,X[i,],right)) #do a horizontal padding
              resized = cv2.resize(square, (112, 112), interpolation=cv2.INTER_NEAREST)  #resize
            # In case image is squared
            else:
              resized = cv2.resize(X[i,], (112, 112), interpolation=cv2.INTER_NEAREST)  #resize






            # Store class
            y[i] = self.labels[ID]

        # the output is transformed into 1-hot encoded vector
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)





  
meteor = DataGenerator(batch_size=10, df=myframe)
myGen = meteor.name_generation(myframe,10)

for el in myGen:
  print(len(el))
  #print(el)
  break




 
