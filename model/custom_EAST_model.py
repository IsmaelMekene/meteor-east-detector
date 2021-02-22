



#load a ResNet101V2 model
meteor2_resnet101v2 = tf.keras.applications.ResNet101V2(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)




# rearrange the model by selecting the needed layers.
meteor_model = tf.keras.Model(inputs=meteor2_resnet101v2.input, outputs=[meteor2_resnet101v2.get_layer('conv1_conv').output, 
                                                                  meteor2_resnet101v2.get_layer('conv2_block3_1_relu').output,
                                                                  meteor2_resnet101v2.get_layer('conv3_block4_1_relu').output,
                                                                  meteor2_resnet101v2.get_layer('conv4_block23_1_relu').output,
                                                                  meteor2_resnet101v2.get_layer('conv5_block3_2_relu').output])
                                                                  
                                                                  
                                                                  
                                                
           

#Store the needed layers in variables according the EAST paper

f1 = meteor2_resnet101v2.get_layer('conv5_block3_2_relu').output  #f1 according to EAST paper
f2 = meteor2_resnet101v2.get_layer('conv4_block23_1_relu').output  #f2 according to EAST paper
f3 = meteor2_resnet101v2.get_layer('conv3_block4_1_relu').output  #f3 according to EAST paper
f4 = meteor2_resnet101v2.get_layer('conv2_block3_1_relu').output  #f4 according to EAST paper
f5 = meteor2_resnet101v2.get_layer('conv1_conv').output  #f5 according to EAST paper


#Follow the steps according the EAST paper

#First green block (h1 ---> h2)
unpool_h1 = UpSampling2D(size=(2, 2), interpolation="nearest")(f1) #unpool the layer in order to make concatenation possible
concat_h1_f2 = Concatenate()([unpool_h1, f2])  #concatenate with the f2 layer
conv1vs1_in_h1 = tf.keras.layers.Conv2D(filters = 128, kernel_size = 1, padding="same", activation="relu")(concat_h1_f2)  #make a (1×1) 2D convolution
conv3vs3_in_h1 = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding="same", activation="relu")(conv1vs1_in_h1) #make a (3×3) 2D convolution

#Second green block (h2 ---> h3)
unpool_h2 = UpSampling2D(size=(2, 2), interpolation="nearest")(conv3vs3_in_h1) #unpool the layer in order to make concatenation possible
concat_h2_f3 = Concatenate()([unpool_h2, f3])  #concatenate with the f3 layer
conv1vs1_in_h2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 1, padding="same", activation="relu")(concat_h2_f3)  #make a (1×1) 2D convolution
conv3vs3_in_h2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding="same", activation="relu")(conv1vs1_in_h2) #make a (3×3) 2D convolution

#Third green block (h3 ---> h4)
unpool_h3 = UpSampling2D(size=(2, 2), interpolation="nearest")(conv3vs3_in_h2) #unpool the layer in order to make concatenation possible
concat_h3_f4 = Concatenate()([unpool_h3, f4])  #concatenate with the f3 layer
conv1vs1_in_h3 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 1, padding="same", activation="relu")(concat_h3_f4)  #make a (1×1) 2D convolution
conv3vs3_in_h3 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding="same", activation="relu")(conv1vs1_in_h3) #make a (3×3) 2D convolution

#Fourth green block (h4 ---> h5)
npool_h4 = UpSampling2D(size=(2, 2), interpolation="nearest")(conv3vs3_in_h3) #unpool the layer in order to make concatenation possible
concat_h4_f5 = Concatenate()([unpool_h4, f5])  #concatenate with the f3 layer
conv1vs1_in_h4 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 1, padding="same", activation="relu")(concat_h4_f5)  #make a (1×1) 2D convolution
conv3vs3_in_h4 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding="same", activation="relu")(conv1vs1_in_h4) #make a (3×3) 2D convolution

#Last green block
conv3vs3_in_h5 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding="same", activation="relu")(conv3vs3_in_h4)

#First blue block (first output layer): SCORE_map
conv1vs1_in_outputlayer = tf.keras.layers.Conv2D(filters = 1, kernel_size = 1, padding="same", activation="sigmoid",name="outputlayer_SCORE")(conv3vs3_in_h5)

#Second blue block (second output layer): QUAD geometry
conv1vs1_in_QUADgeometry = tf.keras.layers.Conv2D(filters = 4, kernel_size = 1, padding="same", activation="relu", name="QUAD_geometry")(conv3vs3_in_h5)


#Building the desired Model
our_meteor_model = tf.keras.Model(inputs=meteor2_resnet101v2.input, outputs=[conv1vs1_in_outputlayer, conv1vs1_in_QUADgeometry])


#Plot the model 
tf.keras.utils.plot_model(
    our_meteor_model, to_file='model.png', show_shapes=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)


