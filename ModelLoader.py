
import tflearn
from tflearn.layers.core import dropout, fully_connected, input_data
from tflearn.layers.estimator import regression


def model_loader(modelname='Models/model1.tflearn', inputs=13, LR=1e-3):

    # Layer 0: generates a 4D tensor
    layer0 = input_data(shape=[None, 1, inputs, 1], name='input')

    # Layer 1
    layer1 = fully_connected(layer0, 32, activation='relu')
    dropout1 = dropout(layer1, 0.8)  # what is dropout?

    # Layer 2
    layer2 = fully_connected(dropout1, 32, activation='relu')
    dropout2 = dropout(layer2, 0.8)

    # Layer 3
    layer3 = fully_connected(dropout2, 32, activation='relu')
    dropout3 = dropout(layer3, 0.8)

    # Layer 4
    layer4 = fully_connected(dropout3, 32, activation='relu')
    dropout4 = dropout(layer4, 0.8)

    # this layer needs to spit out the number of categories we are looking for.
    softmax = fully_connected(dropout4, 2, activation='softmax')

    network = regression(softmax, optimizer='Adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_verbose=0)

    model.load(modelname)

    return(model)
