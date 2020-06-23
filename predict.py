#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
import cv2
import numpy as np


# In[2]:


# Load resnet model
path_model = 'Resnet_Emotion.h5'
model = load_model(path_model)


# In[3]:


# init values
prototxt = 'deploy.prototxt.txt'
caffe_model = 'res10_300x300_ssd_iter_140000.caffemodel'

#load model
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)


# In[24]:


# read image
path_image = 'image.jpg'
image = cv2.imread(path_image)
'''
cv2.imshow('carot', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


# In[26]:


(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass blob through the network and get the detection
net.setInput(blob)
detections = net.forward()
detections = detections[0][0]

detections = np.array(sorted(detections, key=lambda x: x[2], reverse=True))

threshold = detections[0, 2]
# filter out weak detections by ensuring the `confidence` is
# greater than the minimum confidence
if threshold > 0.5:
    # compute the (x, y)-coordinates of the bounding box for the object
    box = detections[0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    
    if (0 <= startX <= w) and (0 <= endX <= w) and (0 <= startY <= h) and (0 <= endY <= h):
        # crop face
        crop_face = image[startY:endY, startX:endX, :]
        
        # reshape image into (64, 64, 3), normalize 
        crop_face = cv2.resize(crop_face, (64, 64))
        crop_face = crop_face/255
        crop_face = np.expand_dims(crop_face, axis=0)
        
        # predict my image
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        result_predict = model.predict(crop_face)[0]
        emotion = emotions[np.argmax(result_predict)]
        print('emotion: ', emotion)
        
        # put emotion into image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, emotion, (startX, startY), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow('carot', image)
        cv2.imwrite('predict.jpg', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# In[ ]:




