# ----- RESNET 50 -----
import h5py
from keras import optimizers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, \
    Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import  Model, load_model
from keras.initializers import glorot_uniform
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basic
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve filters
    F1, F2, F3 = filters

    # Save X_shortcut
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters= F1, kernel_size= (1, 1), strides= (1, 1), padding= 'valid',\
               name= conv_name_base + '2a', kernel_initializer= glorot_uniform(0))(X)
    X = BatchNormalization(axis= 3, name= bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters= F2, kernel_size= (f, f), strides= (1, 1), padding= 'same',\
               name= conv_name_base + '2b', kernel_initializer= glorot_uniform(0))(X)
    X = BatchNormalization(axis= 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', \
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: add shortcut to main path, pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X

# ----- THE CONVOLUTIONAL BLOCK -----

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve filters
    F1, F2, F3 = filters

    # Save the shortcut
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters= F1, kernel_size= (1, 1), strides= (s, s), padding= 'valid',\
               name= conv_name_base + '2a', kernel_initializer= glorot_uniform(0))(X)
    X = BatchNormalization(axis= 3, name= bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters= F2, kernel_size= (f, f), strides= (1, 1), padding= 'same',\
               name = conv_name_base + '2b', kernel_initializer= glorot_uniform(0))(X)
    X = BatchNormalization(axis= 3, name= bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', \
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Shortcut path
    X_shortcut = Conv2D(filters= F3, kernel_size= (1, 1), strides= (s, s), padding= 'valid', \
                        name= conv_name_base + '1', kernel_initializer= glorot_uniform(0))(X_shortcut)
    X_shortcut = BatchNormalization(axis= 3, name= bn_name_base + '1')(X_shortcut)

    # Final step
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X

# ----- RESNET MODEL 50 LAYERS -----
def ResNet50(input_shape = (64, 64, 3), classes = 7):
    '''
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    '''

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero padding
    X = ZeroPadding2D(padding= (3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters= 64, kernel_size= (7, 7), strides= (2, 2), name= 'conv1', kernel_initializer= glorot_uniform(0))(X)
    X = BatchNormalization(axis= 3, name= 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size= (3, 3), strides= (2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f= 3, filters= [64, 64, 256], stage= 2, block= 'a', s= 1)
    X = identity_block(X, f= 3, filters= [64, 64, 256], stage= 2, block= 'b')
    X = identity_block(X, f= 3, filters= [64, 64, 256], stage= 2, block= 'c')

    # Stage 3
    X = convolutional_block(X, f= 3, filters=[128, 128, 512], stage= 3, block= 'a', s= 2)
    X = identity_block(X, f= 3, filters= [128, 128, 512], stage= 3, block= 'b')
    X = identity_block(X, f= 3, filters= [128, 128, 512], stage= 3, block= 'c')
    X = identity_block(X, f= 3, filters= [128, 128, 512], stage= 3, block= 'd')

    # Stage 4
    X = convolutional_block(X, f= 3, filters=[256, 256, 1024], stage= 4, block= 'a', s= 2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2))(X)

    X = Flatten()(X)

    # Fully connected
    X = Dense(classes, activation= 'softmax', name= 'fc' + str(classes), kernel_initializer= glorot_uniform(0))(X)

    # Create model
    model = Model(inputs= X_input, outputs= X, name= 'ResNet50')

    return model

# read IMAGE file from h5 file
h5f = h5py.File('image_train.h5', 'r')
array_image = h5f['dataset'][:]
h5f.close()

# read LABEL file from h5 file
h5f = h5py.File('label_train.h5', 'r')
array_label = h5f['dataset'][:]
h5f.close()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#convert to one hot vector
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

array_label = convert_to_one_hot(array_label, 7).T

# shuffle and normalize data
from sklearn.utils import shuffle
import numpy as np
array_image, array_label = shuffle(array_image, array_label)
array_image = array_image/255

model = ResNet50(input_shape= (64, 64, 3), classes= 7)

sgd = optimizers.SGD(lr=0.01, decay=0.0, momentum=0.05, nesterov=True)

model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.fit(array_image, array_label, epochs = 1, batch_size = 256)

# evaluate the model
scores = model.evaluate(array_image, array_label, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("Resnet_Emotion_test.h5")
print("Saved model to disk")
print('FINISH')