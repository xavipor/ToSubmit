import scipy.io as sio
import math
import numpy as np
import h5py
import cPickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
from Bachitazion import Bachitazion
import pickle
from os import listdir
from os.path import isfile,join
from os import walk
import os



##data_path = whole_volume_path + str(74) + '.mat'
#data_set = np.transpose(np.array(h5py.File(data_path)['patchFlatten']))
#image =  data_set.reshape((data_set.shape[0],10,16,16,1))
def save_to_file(filename, object):
    """ Save object to file
    """
    f = open(filename + '.pckl', 'wb')
    pickle.dump(object, f)
    f.close()

def createPlaceHolders(n_H,n_W,n_C,n_D,n_Y):
    """
    Create placeholder for the session

    Arguments:
        n_H -- Height of the input image
        n_W -- Width of the input image
        n_D -- Depth of the input image
        n_C -- Channels of the input image
        n_y -- number of classes
    """
    X  = tf.placeholder(tf.float32,[None,n_D,n_H,n_W,n_C],name="X")
    Y = tf.placeholder(tf.float32,[None,n_Y],name = "Y")
    return X,Y

def initializeWeights(preTrained = True,path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf2/'):
    if preTrained:
        path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/WeightsTrainedSN/'
        W_L0 = np.load(path+'W0.npy')
        b_L0 = np.load(path+'b0.npy')

        W_L1 = np.load(path+'W1.npy')
        b_L1 = np.load(path+'b1.npy')

        W_L2 = np.load(path+'W2.npy')
        b_L2 = np.load(path+'b2.npy')

        W_L3 = np.load(path+'W3.npy')
        b_L3 = np.load(path+'b3.npy')

        W_L4 = np.load(path+'W4.npy')
        b_L4 = np.load(path+'b4.npy')

        W0 = tf.Variable(W_L0, name="W0",trainable = True)
        W1 = tf.Variable(W_L1, name="W1",trainable =True)
        W2 = tf.Variable(W_L2, name="W2",trainable = True)

        W3a = tf.Variable(W_L3, name="W3a")
#        W3b = tf.reshape(W3a,shape=[512,150])
        W4a = tf.Variable(W_L4, name="W4a")
#        W4b = tf.reshape(W4a,shape=[150,2])


        W3 = tf.Variable(W3a, name="W3")
        W4 = tf.Variable(W4a, name="W4")

        b0 = tf.Variable(b_L0, name ="b0",trainable = True)
        b1 = tf.Variable(b_L1, name ="b1",trainable = True)
        b2 = tf.Variable(b_L2, name ="b2",trainable= True)
        b3 = tf.Variable(b_L3, name ="b3",trainable= True)
        b4 = tf.Variable(b_L4, name ="b4",trainable= True)


    #Would be perfect to define another part if it was created from scratch, but still to do

    parameters={"W0":W0,"b0":b0,"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4}

    return parameters

def computeCost(Z5,Y,beta,W3,W4,W1,W2):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5,labels=Y))
    reg = tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)  + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W1)
    costa = tf.reduce_mean(cost + beta*reg)
    return costa

def forward_propagation(X,parameters):
    #This could have been done better defininf a Customized Convolution Layer and the same for flatten layer.
    #I mean, the result is the same but is more structured.

    W_L0 = parameters["W0"]
    b_L0 = parameters["b0"]

    W_L1 = parameters["W1"]
    b_L1 = parameters["b1"]

    W_L2 = parameters["W2"]
    b_L2 = parameters["b2"]

    W_L3 = parameters["W3"]
    b_L3 = parameters["b3"]

    W_L4 = parameters["W4"]
    b_L4 = parameters["b4"]


    # Retrieve the parameters from the dictionary "parameters"

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME
    Z1a =tf.nn.conv3d(X,W_L0,strides=[1,1,1,1,1],padding='VALID')
    Z1b = tf.nn.bias_add(Z1a,b_L0)
    Z1c= tf.nn.max_pool3d(Z1b,ksize=(1,2,2,2,1),strides=(1,2,2,2,1),padding='VALID')
    A1 = tf.nn.relu(Z1c)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    Z2a = tf.nn.conv3d(A1,W_L1,strides=[1,1,1,1,1],padding='VALID')
    Z2b = tf.nn.bias_add(Z2a,b_L1)
    A2 = tf.nn.relu(Z2b)
    P2 = tf.contrib.layers.flatten(A2)
	

    Z3 = tf.add(tf.matmul(P2,W_L2),b_L2)
    A3 = tf.nn.relu(Z3)

    Z4 = tf.add(tf.matmul(A3,W_L3),b_L3)
#   Z4 =tf.nn.conv3d(A3,W_L3,strides=[1,1,1,1,1],padding='VALID')
#   Z4 = tf.nn.bias_add(Z4,b_L3)
    A4 = tf.nn.relu(Z4)

    #Z5=tf.nn.conv3d(A4,W_L4,strides=[1,1,1,1,1],padding='VALID')
    #Z5 = tf.nn.bias_add(Z5,b_L4)
    Z5 =tf.add(tf.matmul(A4,W_L4),b_L4)
    A5 = tf.nn.softmax(Z5)
    return A5

def forward(X,parameters):
    #This could have been done better defininf a Customized Convolution Layer and the same for flatten layer.
    #I mean, the result is the same but is more structured.

    W_L0 = parameters["W0"]
    b_L0 = parameters["b0"]

    W_L1 = parameters["W1"]
    b_L1 = parameters["b1"]

    W_L2 = parameters["W2"]
    b_L2 = parameters["b2"]

    W_L3 = parameters["W3"]
    b_L3 = parameters["b3"]

    W_L4 = parameters["W4"]
    b_L4 = parameters["b4"]


    # Retrieve the parameters from the dictionary "parameters"

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME
    Z1a =tf.nn.conv3d(X,W_L0,strides=[1,1,1,1,1],padding='VALID')
    Z1b = tf.nn.bias_add(Z1a,b_L0)
    Z1c= tf.nn.max_pool3d(Z1b,ksize=(1,2,2,2,1),strides=(1,2,2,2,1),padding='VALID')
    A1 = tf.nn.relu(Z1c)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    Z2a = tf.nn.conv3d(A1,W_L1,strides=[1,1,1,1,1],padding='VALID')
    Z2b = tf.nn.bias_add(Z2a,b_L1)
    A2 = tf.nn.relu(Z2b)
    P2 = tf.contrib.layers.flatten(A2)


    Z3 = tf.add(tf.matmul(P2,W_L2),b_L2)
    A3 = tf.nn.relu(Z3)

    Z4 = tf.add(tf.matmul(A3,W_L3),b_L3)
#   Z4 =tf.nn.conv3d(A3,W_L3,strides=[1,1,1,1,1],padding='VALID')
#   Z4 = tf.nn.bias_add(Z4,b_L3)
    A4 = tf.nn.relu(Z4)

    #Z5=tf.nn.conv3d(A4,W_L4,strides=[1,1,1,1,1],padding='VALID')
    #Z5 = tf.nn.bias_add(Z5,b_L4)
    Z5 =tf.add(tf.matmul(A4,W_L4),b_L4)
    A5 = tf.nn.softmax(Z5)
    return Z5



def train(learning_rate=0.0001,num_epochs =5000,beta=0):
#    myPathModel = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModel/Origina'
#    if not os.path.exists(myPathModel):
#        os.makedirs(myPathModel)
#    myBatchGenerator = Bachitazion(sizeOfBatch=2048,pathT='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-co$
#    myBatchGenerator = Bachitazion(sizeOfBatch=100,pathT='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-co$
    err_val={}
    acc_val={}
    err_train={}
    acc_train={}
    X, Y = createPlaceHolders(20,20,1,16,2)
    parameters = initializeWeights()
    outputs = forward_propagation(X,parameters)
    feo = forward(X,parameters)
    cost = computeCost(outputs, Y, beta,parameters["W3"],parameters["W4"],parameters["W1"],parameters["W2"])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
 # Calculate the correct predictions
    correctPrediction = tf.equal(tf.argmax(outputs,1), tf.argmax(Y,1))
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    #print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=None)
    results_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/result/final_prediction/'
    with tf.Session() as sess:
        sess.run(init)
        datapath = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/result/test_set_cand/'
        files = os.listdir(datapath)
        n_cases = len(files)
        print ('n_cases:',files)
        for cs in range(n_cases):
            true_list=[]
            overallPrediction = np.array([])
            case = cs + 1
            set_x = np.array(h5py.File(datapath + str(case) + '_patches.mat')['test_set_x'])
#            pdb.set_trace()
            set_x2 = np.transpose(set_x)
#            set_x2 = np.transpose(set_x) - np.mean(set_x)
            print ('predicting {0} subject, contains {1} candidates...'.format(case, set_x2.shape[0]))
#            layer0_input = set_x.reshape((set_x.shape[0],1,16,20,20))
#            layer0_input2 = layer0_input.transpose(0,2,3,4,1)
#    X = X = tf.placeholder(tf.float32,[None,n_D,n_H,n_W,n_C],name="X")
            patch=set_x2.reshape([set_x2.shape[0],20,20,16,1],order='F').copy()
           # layer0_input = set_x2.reshape((set_x2.shape[0],1,16,20,20))
            layer0_input2 = patch.transpose(0,3,2,1,4)

            scores= sess.run(outputs,{X:layer0_input2})
            value_Z5 = sess.run (feo,{X:layer0_input2})
#	    pdb.set_trace()
#            for n,e in enumerate(value_Z5):
#                if e == 1: true_list.append(n+1)
            sio.savemat(results_path + str(case)+'_prediction.mat',{'prediction':scores})
        sess.close()

train()

