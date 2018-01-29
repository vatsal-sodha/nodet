#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

#DEEP NETS BASED ON PIXELS AS FEATURES

'''The below code is an implementation of deep neural nets which takes pixels as input features'''

#Imports
from __future__ import division #To perform float divisons
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from glob import * #Glob for the functions to read files and file paths
from PIL import Image #Image Processing Library
import pandas as pd #Pandas is used to manipulate data frames



#Loading Train and Test Data 
def load_data(j):

  if j==0:
    file_list = glob('Train_1to10_Data/Train_1to10_Data/*')
    data = np.zeros((11658,4096))
  if j==1:
    file_list = glob('Test_1to10_Data/Test_1to10_Data/*')
    data = np.zeros((3300,4096))

  i = 0
  for img_file in file_list:
    img_file = Image.open(img_file)
    arr = np.array(img_file)
    arr = arr.flatten()
    print(img_file.filename)
    data[i] = arr
    i = i + 1
  data = np.array(data)
  return np.transpose(data)

#Loading Train and Test Labels
def load_labels(j):
  if j==0:
    df = pd.read_csv('Train_1to10_Lables.csv')
  if j==1:
    df = pd.read_csv('Test_1to10_Lables.csv')
  labels = df['Class'].tolist()
  labels = np.array(labels)
  return labels

#Initialize Parameters - Weights and Bias through Xavier Initialization
def initialize_parameters():
    
    W1 = tf.get_variable("W1", [25, 4096], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [2, 12], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [2, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters




#Forward Prop
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

#Cost
def compute_cost(Z3, Y):
   
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

#For Randomn mini batches
def random_mini_batches(X, Y, mini_batch_size = 64):

    
    m = X.shape[1]             
    mini_batches = []
 
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

#Coversion to One_Hot
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#Prediction
def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [4096, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
    print('---------------\n',prediction)
    print(type(prediction))
    print(prediction['10'])
    return prediction

#Forward Prop for prediction
def forward_propagation_for_predict(X, parameters):
   
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

#Final Model
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 100, minibatch_size = 32, print_cost = True):
   
    
    ops.reset_default_graph()        
    (n_x, m) = X_train.shape                          
    n_y = Y_train.shape[0]                            
    costs = []                                        
    
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)
 
    cost = compute_cost(Z3, Y)
  
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
               	epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        


        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        print("--------------")
        check = tf.contrib.util.constant_value(Z3)
        print(check[100])

        #with tf.Session():
        #	print('Ze:', tf.Tensor.eval(tf.argmax(Z3)))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #with tf.Session():
        #	print('Confusion Matrix:', tf.Tensor.eval(confusion))


        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters




# Main Function
def main():

	X_train_orig = load_data(0) 
	X_test_orig  = load_data(1)

	print X_train_orig.shape

	Y_train_orig = load_labels(0)
	Y_test_orig  = load_labels(1)


	# Normalize image vectors
	X_train = X_train_orig / 255.
	X_test = X_test_orig / 255.

	# Convert training and test labels to one hot matrices
	Y_train = convert_to_one_hot(Y_train_orig, 2)
	Y_test = convert_to_one_hot(Y_test_orig, 2)

	print Y_train.shape


	#Run the model
	parameters = model(X_train, Y_train, X_test, Y_test)


#Calling Main function

main()