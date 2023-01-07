import cv2
import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model('food_classifier.h5')
img_size = 200
input_shape = [img_size,img_size,1]

with open('objects.pickle','rb') as objs:
    categories = pickle.load(objs)

def make_prediction(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(img_size,img_size))
    img = img.reshape(img_size,img_size,1)
    img = img.reshape(1,img_size,img_size,1)
    img = img/255
    pred = model.predict(img)
    return categories.get(np.argmax(pred)) if max(pred[0]) > 0.5 else 'Can\'t identify'