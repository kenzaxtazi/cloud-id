#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:28:06 2018

@author: kenzatazi
"""

import numpy as np
import os
from random import shuffle
import cv2
from tqdm import tqdm#percentage bar for tasks.
import matplotlib.pyplot as plt

TRAIN_DIR = '/Users/kenzatazi/Downloads/catsanddogs/train'
TEST_DIR = '/Users/kenzatazi/Downloads/catsanddogs/test'
IMG_SIZE = 50
LR = 1e-3

## Format ??? # FIXME   

MODEL_NAME = 'dogsvscats.model'.format(LR, '2conv-basic') 
# just so we remember which saved model is which, sizes must matchca



###### PREPROCESSING

def create_label(image_name):
    """ Creates an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])
    
def create_training_data():
    """ 
    Reformats and grayscales the images so they are all the same. The function
    also adds a label and an image number before saving the images in a random 
    order as an npy file.
    """ 
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    """
    Reformats and grayscales the images so they are all the same.The function
    also adds an image number before saving the images in a random order,
    as an npy file.
    """ 
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# If dataset is not created:
train_data = create_training_data()
test_data = create_test_data()
# If dataset already created :
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')





#### MACHINE LEARNING 

"""
TFlearn is a modular and transparent deep learning library built on top of 
Tensorflow. It was designed to provide a higher-level API to TensorFlow in 
order to facilitate and speed-up experimentations, while remaining fully 
transparent and compatible with it.
"""

"""
conv_2d: takes a 4D tensor and spits out a 4D tensor   

The number of filters: 
    
The size of the filters: Common filter shapes found in the literature vary 
greatly, and are usually chosen based on the dataset. The challenge is, thus, 
to find the right level of granularity so as to create abstractions at the 
proper scale, given a particular dataset.

The activation: the function which maps the output of a layer to a map with a
smaller number of dimensions.

'relu' is the abbreviation of Rectified Linear Units. This layer applies the 
non-saturating activation function f(x)=max(0,x). It increases the nonlinear 
properties of the decision function and of the overall network without 
affecting the receptive fields of the convolution layer. This function is 
prefered to other non linearising functions because of its speed.

"""

"""
max_pool_2d: 
Convolutional networks may include local or global pooling layers which combine
the outputs of neuron clusters at one layer into a single neuron in the next 
layer. Max pooling uses the maximum value from each of a cluster of neurons at 
the prior layer. 

"""

"""
input_data:
This layer is used for inputting (aka. feeding) data to a network. A TensorFlow
placeholder will be used if it is supplied, otherwise a new placeholder will be
created with the given shape.
"""


"""
dropout:
Outputs the input element scaled up by 1 / keep_prob. The scaling is so that 
the expected sum is unchanged.

keep_prob : a float representing the probability that each element is kept.
"""


"""
fully connected:
Fully connected layers connect every neuron in one layer to every neuron in 
another layer. It is in principle the same as the traditional multi-layer 
perceptron neural network. 

n_units= number of units in the layer

"""


"""
regression :
The regression layer is used in TFLearn to apply a regression (linear or 
logistic) to the provided input. It requires to specify a TensorFlow gradient 
descent optimizer 'optimizer' that will minimize the provided loss function 
'loss' (which calculate the errors). A metric can also be provided, to evaluate
the model performance.

'Adam' optimiser : A Method for Stochastic Optimization. Diederik Kingma, 
Jimmy Ba. ICLR 2015 (Learning_rate=0.001, beta1=0.9, beta2=0.999, 
epsilon=1e-08, use_locking=False). 

'categorical_crossentropy' loss: this means we want to minimise categorical 
cross entropy  i.e. how much the model classifies an image as dog and cat. 
    
"""


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Layer 0: generates a 4D tensor
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

# Layer 1
convnet = conv_2d(convnet, nb_filter=32, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet,kernel_size=5)

# Layer 2
convnet = conv_2d(convnet, nb_filter=64, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 3
convnet = conv_2d(convnet, nb_filter=128, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 4
convnet = conv_2d(convnet, nb_filter=64, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 5
convnet = conv_2d(convnet, nb_filter=32, filter_size=5, activation='relu')
convnet = max_pool_2d(convnet, kernel_size=5)

# Layer 6
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, keep_prob=0.8)

# Layer 7
convnet = fully_connected(convnet, n_units=2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, \
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
# we want to use tensorboard to visualise the training but we haven't got it 
# running yet

if os.path.exists('/Users/kenzatazi/Documents/University/Year 4/Msci Project/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')



### UNPACK SAVED DATA


"""
model.fit:
Trains model, feeding X_inputs and Y_targets to the network.
"""

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, \
          validation_set=({'input': test_x}, {'targets': test_y}), \
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)


### TESTING ON UNLABELLED DATA


"""
model.predict:
Gives model prediction for given input data.

"""

fig=plt.figure(figsize=(16, 12))

for num, data in enumerate(test_data[:16]):
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: 
        str_label='Dog'
    else:
        str_label='Cat'
        
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


