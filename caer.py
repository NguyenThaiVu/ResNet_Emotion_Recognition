# ----- CAER -----
import keras
from keras import optimizers
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, \
    Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Concatenate, Reshape, Multiply, concatenate, \
    Lambda
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.initializers import glorot_uniform
import keras.backend as K

K.set_learning_phase(1)
import cv2
import os


# Define a Face encoding stream (static image)
def face_encoding_stream(IF, f=3, filters=[32, 64, 128, 256, 256]):
    '''
    Implement face encoding stream
    :param IF: input image face cropped
    :param f: size of kernel, in this case (3x3)
    :param filters: list of integers, defining the number of filters in the CONV layers \
                    [32, 64, 128, 256, 256]
    :return: output of face encoding
    '''
    # Get filters
    F1, F2, F3, F4, F5 = filters
    # First conv
    XF = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        IF)
    XF = BatchNormalization(axis=3)(XF)
    XF = Activation('relu')(XF)
    XF = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XF)

    # Second conv
    XF = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XF)
    XF = BatchNormalization(axis=3)(XF)
    XF = Activation('relu')(XF)
    XF = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XF)

    # Third conv
    XF = Conv2D(filters=F3, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XF)
    XF = BatchNormalization(axis=3)(XF)
    XF = Activation('relu')(XF)
    XF = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XF)

    # Forth conv
    XF = Conv2D(filters=F4, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XF)
    XF = BatchNormalization(axis=3)(XF)
    XF = Activation('relu')(XF)
    XF = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XF)

    # Fifth conv
    XF = Conv2D(filters=F5, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XF)
    XF = BatchNormalization(axis=3)(XF)
    XF = Activation('relu')(XF)

    XF = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(XF)

    return XF


# Define context encoding stream (static image)
def context_encoding_stream(IC, f=3, filters=[32, 64, 128, 256, 256]):
    '''
    Implement context_encoding stream
    :param IC: input context (face hidden)
    :param f: kernel size
    :param filters: list of integers, defining the number of filters in the CONV layers [32, 64, 128, 256, 256]
    :return: output of context encoding
    '''
    # Get filters
    F1, F2, F3, F4, F5 = filters

    # First conv
    XC = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        IC)
    XC = BatchNormalization(axis=3)(XC)
    XC = Activation('relu')(XC)
    XC = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XC)

    # Second conv
    XC = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XC)
    XC = BatchNormalization(axis=3)(XC)
    XC = Activation('relu')(XC)
    XC = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XC)

    # Third conv
    XC = Conv2D(filters=F3, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XC)
    XC = BatchNormalization(axis=3)(XC)
    XC = Activation('relu')(XC)
    XC = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XC)

    # Forth conv
    XC = Conv2D(filters=F4, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XC)
    XC = BatchNormalization(axis=3)(XC)
    XC = Activation('relu')(XC)
    XC = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(XC)

    # Fifth conv
    XC = Conv2D(filters=F5, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XC)
    XC = BatchNormalization(axis=3)(XC)
    XC = Activation('relu')(XC)

    XC = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(XC)

    return XC


# Define adaptive fusion network (static image)
def adaptive_fusion_network(XF, XC):
    '''
    XF: face_stream.output
    XC: context_stream.output
    '''

    # --- First step, adaptive network for face encoding ---
    XF_copy = XF
    XF = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XF)
    XF = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(XF)

    # --- Second step, adaptive network for context encoding ---
    XC_copy = XC
    XC = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(
        XC)
    XC = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(XC)

    # Softmax
    # ...
    # __________________

    # Fusion

    face_concatenation = Multiply()([XF, XF_copy])
    print('face_concatenation shape: ', face_concatenation.shape)

    context_concatenation = Multiply()([XC, XC_copy])
    print('context_concatenation shape: ', context_concatenation.shape)

    XA = concatenate([face_concatenation, context_concatenation])
    print('After concat shape: ', XA.shape)

    XA = Flatten()(XA)

    # Final classifier
    XA = Dense(128, activation='relu', kernel_initializer=glorot_uniform(0))(XA)
    XA = Dense(7, activation='softmax', kernel_initializer=glorot_uniform(0))(XA)

    return XA


# ----- CAER MODEL -----
def CAER(input_shape=(96, 96, 3), classes=7):
    # IF: image face
    # IC: image context

    # Define the input as a tensor with shape input_shape
    # Notice: X_input is tensor, so shape[0] = ? - is a length of training set

    # Face encoding stream
    IF = Input(shape=input_shape)
    XF = face_encoding_stream(IF, f=3, filters=[32, 64, 128, 256, 256])
    face_stream = Model(IF, XF)

    # Context encoding stream
    IC = Input(shape=input_shape)
    XC = context_encoding_stream(IC, f=3, filters=[32, 64, 128, 256, 256])
    context_stream = Model(IC, XC)

    # Define Fusion network
    XA = adaptive_fusion_network(face_stream.output, context_stream.output)

    # Create model
    model = Model(inputs=[IF, IC], outputs=XA)

    return model


model = CAER(input_shape=(96, 96, 3), classes=7)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=[image_face_train, image_context_train], y=label_face_train, epochs=20, batch_size=128)

# evaluate the model
scores = model.evaluate(image_train_flat, label_face_train, verbose=0)
print("%s ---:--- %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# save model and architecture to single file
model.save("model_caer.h5")
print("Saved model to disk")
print('FINISH')