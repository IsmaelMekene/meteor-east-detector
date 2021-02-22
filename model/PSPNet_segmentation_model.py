



!pip install segmentation-models

%env SM_FRAMEWORK=tf.keras

from segmentation_models import PSPNet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss

BACKBONE = 'resnet152'
preprocess_input = get_preprocessing(BACKBONE)
pspnet_model = PSPNet(backbone_name=BACKBONE, classes=1,input_shape=(480,480, 3), activation='sigmoid', encoder_weights='imagenet')

#pspnet_model.summary()



#optimizer

optimizer_adam = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
    
    
    
