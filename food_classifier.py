
import os
import random
import cv2
import numpy as np
import tensorflow as tf
import pickle

pth = 'Fast Food Classification V2'

categories = {i:j for i,j in enumerate(os.listdir(f'{pth}/Train'))}
img_size = 200
batch_size = 32
input_shape = [img_size,img_size,1]


def load_data(path):
    data = list()
    for idx, folder in categories.items():
        folder_path = os.path.join(path,folder)
        images = os.listdir(folder_path)
        images = [os.path.join(folder_path,image) for image in images]
        for image in images:
            data.append([image,idx])
    return data


train = load_data(f'{pth}/Train')
test = load_data(f'{pth}/Test')
evall = load_data(f'{pth}/Valid')

with open('objects.pickle','wb') as objs:
    pickle.dump(categories,objs)

random.shuffle(train)
random.shuffle(test)
random.shuffle(evall)
random.shuffle(train)
random.shuffle(test)
random.shuffle(evall)


class DataSequence(tf.keras.utils.Sequence):

    def __init__(self,data,batch_size):
        self.data = data
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self,idx):
        batch = self.data[idx * self.batch_size : (idx + 1) * self.batch_size]
        return self.data_gen(batch)

    def data_gen(self,data):
        images,labels = list(),list()
        for image,label in data:
            img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(img_size,img_size))
            img = img.reshape(*input_shape)
            img = img/255
            images.append(img)
            labels.append(label)
        return np.array(images),np.array(labels)

train_data = DataSequence(train,batch_size)
test_data = DataSequence(test,1)
eval_data = DataSequence(evall,batch_size)

# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal',name='Random_Flip'),
#     tf.keras.layers.experimental.preprocessing.RandomRotation(0.1,name='Random_Rotation'),
#     tf.keras.layers.experimental.preprocessing.RandomZoom(0.1,name='Random_Zoom'),
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[200,200,1],name='Input_Layer'),
    # data_augmentation,

    # tf.keras.layers.Conv2D(32,kernel_size=(3,3),name='Conv_Layer_1'),
    # tf.keras.layers.Activation('relu',name='Conv_Relu_1'),
    # tf.keras.layers.MaxPool2D((3,3),name='Conv_Max_Pool_1'),

    tf.keras.layers.Conv2D(64,kernel_size=(3,3),name='Conv_Layer_2'),
    tf.keras.layers.Activation('relu',name='Conv_Relu_2'),
    tf.keras.layers.MaxPool2D((3,3),name='Conv_Max_Pool_2'),

    tf.keras.layers.Conv2D(128,kernel_size=(3,3),name='Conv_Layer_3'),
    tf.keras.layers.Activation('relu',name='Conv_Relu_3'),
    tf.keras.layers.MaxPool2D((3,3),name='Conv_Max_Pool_3'),

    tf.keras.layers.Conv2D(256,kernel_size=(3,3),name='Conv_Layer_4'),
    tf.keras.layers.Activation('relu',name='Conv_Relu_4'),
    tf.keras.layers.MaxPool2D((3,3),name='Conv_Max_Pool_4'),

    tf.keras.layers.Flatten(name='Flatten'),

    tf.keras.layers.Dense(256,name='Dense_Layer_1'),
    tf.keras.layers.Activation('relu',name='Dense_Relu_1'),
    
    tf.keras.layers.Dense(512,name='Dense_Layer_2'),
    tf.keras.layers.Activation('relu',name='Dense_Relu_2'),

    # tf.keras.layers.Dropout(0.2,name='Dropout_1'),

    # tf.keras.layers.Dense(1024,name='Dense_Layer_3'),
    # tf.keras.layers.Activation('relu',name='Dense_Relu_3'),
    
    tf.keras.layers.Dense(len(categories),name='Dense_Layer_2__Classifier'),
    tf.keras.layers.Activation('softmax',name='Dense_Softman_1__Classifier_Activation'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)


with tf.device('/gpu:0'):
    history = model.fit(train_data,epochs=500,validation_data=eval_data,verbose=1)

model.save('food_classifier.h5')