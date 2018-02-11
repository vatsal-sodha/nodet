#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

#CONV NET MODEL

'''The below code trains and tests a CNN Model'''

#Imports
from __future__ import division #To perform float divisons
from glob import * #Glob for the functions to read files and file paths
from PIL import Image #Image Processing Library
import numpy as np #Numpy to manipulate n-dimensional arrays
import tensorflow as tf #TensorFlow - ConvNet and DeepNet Functions
import pandas as pd #Pandas is used to manipulate data frames

#TensorFlow function to display information about the version and other details
tf.logging.set_verbosity(tf.logging.INFO)
ratio = 3
# I took the following architecture from the Tensor Flow's CNN-MNIST tutorial as it worked well on MNIST dataset

def cnn_model_fn(features, labels, mode):

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # DropOut - To reduce Over Fitting of the data
  #dropout = tf.layers.dropout(inputs=pool2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Convolutional Layer #2
  conv3 = tf.layers.conv2d(inputs=pool2,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
  # Pooling Layer #2
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  # Convolutional Layer #4

  conv4 = tf.layers.conv2d(inputs=pool3,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

  # Dense Layer
  pool3_flat = tf.reshape(conv4, [-1, 8 * 8 * 64])
  dense = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)

  # DropOut - To reduce Over Fitting of the data
  dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Final Layer - 2 Units - For nodules and non-nodules
  logits = tf.layers.dense(inputs=dropout, units=2)

  # Predictions while training the data set
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Print Probabilities of the two classes
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#Loading the Train and Test data that are prepared by me in the data_prep.py
def load_data(j):

  if j==0:
    file_list = glob('../data/Simple_CNN_Data/Train_Data/*')
    train_data = np.zeros((2198,4096))
  if j==1:
    file_list = glob('../data/Simple_CNN_Data/Test_Data/*')
    train_data = np.zeros((520,4096))

  i = 0
  for img_file in file_list:
    img_file = Image.open(img_file)
    arr = np.array(img_file)
    arr = arr.flatten()
    print(img_file.filename)
    train_data[i] = arr
    i = i + 1
  return train_data

#Loading the Train and Test labels that are prepared by me in the data_prep.py
def load_labels(j):
  if j==0:
    df = pd.read_csv('../data/Simple_CNN_Data/Train_Lables.csv')
  if j==1:
    df = pd.read_csv('../data/Simple_CNN_Data/Test_Lables.csv')
  train_labels = df['Class'].tolist()
  train_labels = np.array(train_labels)
  return train_labels


def getClasses(x):
  pred=[]
  for i in range(len(x)):
    pred.append(x[i]['classes'])
  return pred

def main(unused_argv):

  # Load train and test data
  train_data = load_data(0)
  train_labels = load_labels(0)

  eval_data = load_data(1)
  eval_labels = load_labels(1)

  # Create the Estimator
  nodet_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/nodet_convnet_sod12_model")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_data = train_data.astype(np.float32)
  print(train_data.dtype)

  # TensorFlow Train Function
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,batch_size=100,num_epochs=None,shuffle=True)

  # Classifier
  nodet_classifier.train(input_fn=train_input_fn,steps=1000,hooks=[logging_hook])
  eval_data = eval_data.astype(np.float32)

  # Test the classifier and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

  eval_results = nodet_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

  pred = list(nodet_classifier.predict(input_fn=eval_input_fn))

  preds = getClasses(pred)
  confusion = tf.confusion_matrix(labels=tf.convert_to_tensor(eval_labels), predictions=tf.convert_to_tensor(preds), num_classes=2)

  with tf.Session():
    print("Confusion: ",tf.Tensor.eval(confusion))

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)

  eval_results = nodet_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


# Run the TF App once the main function is called
if __name__ == "__main__":
  tf.app.run()