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
  conv1 = tf.layers.conv2d(inputs=input_layer,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

  # Pooling Layer #2
  # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # DropOut - To reduce Over Fitting of the data
  dropout1 = tf.layers.dropout(inputs=conv2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Convolutional Layer #2
  conv3 = tf.layers.conv2d(inputs=dropout1,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
  # Pooling Layer #2
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  # Convolutional Layer #4
  dropout2 = tf.layers.dropout(inputs=pool3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  conv4 = tf.layers.conv2d(inputs=dropout2,filters=64,kernel_size=[3, 3],padding="same",activation=tf.nn.relu)

  # Dense Layer
  pool3_flat = tf.reshape(conv4, [-1, 16 * 16 * 64])
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # DropOut - To reduce Over Fitting of the data
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

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
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
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
def load_sampled_data_(j):
  trainExamples=1630
  testExamples=1088
  if j==0:
    iterations=trainExamples
  else:
    iterations=testExamples

  if j==0:
    file_list = glob('../data/OSDataSets/OSTrain_0.7_Neg_0.5_Data/*')
    train_data = np.zeros((trainExamples,4096))
  if j==1:
    file_list = glob('../data/OSDataSets/OSTest_0.7_Neg_0.5_Data/*')
    train_data = np.zeros((testExamples,4096))

  i = 0
  for img_file in file_list[0:iterations]:
    img_file = Image.open(img_file)
    arr = np.array(img_file)
    arr = arr.flatten()
    print(img_file.filename)
    train_data[i] = arr
    i = i + 1
  negative=0
  positive=0
  # df=pd.DataFrame(columns=['file_name','Class'])
  labels=[0]*iterations
  filenames=["0"]*iterations

  i=0
  for  filename in file_list[0:iterations]:
    filenames[i]=filename
    if "_positive_" in filename:
      positive=positive+1
      labels[i]=1
    elif "_negative_" in filename:
      negative=negative+1
    i=i+1
  labels = np.array(labels)
  if j==0:
    return train_data, labels,positive,negative,filenames
  elif j==1:
    return train_data,labels,filenames



def getClasses(x):
  pred=[]
  for i in range(len(x)):
    pred.append(x[i]['classes'])
  return pred
def getPreds(filenames,eval_labels,predictions):
  fps=[]
  tps=[]
  tns=[]
  fns=[]
  if len(filenames)==len(eval_labels)==len(predictions):
    for i in range(len(filenames)):
      if eval_labels[i]==0 and predictions[i]==1:
        fps.append(filenames[i])
      elif eval_labels[i]==1 and predictions[i]==1:
        tps.append(filenames[i])
      elif eval_labels[i]==0 and predictions[i]==0:
        tns.append(filenames[i])
      elif eval_labels[i]==1 and predictions[i]==0:
        fns.append(filenames[i])
    return tps,fps,tns,fns
  else:
    print("Incorrect Length")
    return tps,fps


def main(unused_argv):

  train_data,train_labels,positive,negative,train_filenames=load_sampled_data_(0)
  # print(train_labels)
  eval_data,eval_labels,test_filenames=load_sampled_data_(1)
  # print(eval_labels)

  print("No of positive examples in training is ",positive)
  print("No of negative examples in training is ",negative)


  # Load train and test data
  # train_data = load_data(0)
  # train_labels = load_labels(0)

  # eval_data = load_data(1)
  # eval_labels = load_labels(1)



  # Create the Estimator
  nodet_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./../../../Models/cnn_model4_OS_PosNeg50:50_TrainTest70:30")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_data = train_data.astype(np.float32)
  print(train_data.dtype)

  # TensorFlow Train Function
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},y=train_labels,batch_size=32,num_epochs=None,shuffle=True)

  output_str=""
  epochs=1
  iterations=1

  model_list=["cnn_model4_OS_PosNeg50:50_TrainTest70:30"]*int((len(eval_labels)/2))
  tps_fps_df=pd.DataFrame(model_list,columns=['Model'])
  for steps in range(iterations):

    # Classifier
    nodet_classifier.train(input_fn=train_input_fn,steps=epochs,hooks=[logging_hook])
    eval_data = eval_data.astype(np.float32)

    # Test the classifier and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

    test_results = nodet_classifier.evaluate(input_fn=eval_input_fn)
    print(test_results)

    pred = list(nodet_classifier.predict(input_fn=eval_input_fn))

    preds = getClasses(pred)
    # print(test_filenames)
    # print(len(test_filenames))


    tps,fps,tns,fns=getPreds(test_filenames,eval_labels,preds)
    print("True positives are ",len(tps))
    print("False positives are ",len(fps))
    print("False negatives are ",len(fns))
    print("True negatives are ",len(tns))

    current_epoch=(steps+1)*epochs
    tps_fps_df['tps_'+str(current_epoch)]=pd.Series(tps)
    tps_fps_df['fps_'+str(current_epoch)]=pd.Series(fps)
    tps_fps_df['tns_'+str(current_epoch)]=pd.Series(tns)
    tps_fps_df['fns_'+str(current_epoch)]=pd.Series(fns)

    # tps_fps_df=pd.DataFrame.from_dict({'Model':model_list,'tps_800':tps,'fps_800':fps},orient='index')
    # df=tps_fps_df.transpose()
    # df.to_csv('../../../tps&fps/'+model_list[0]+'.csv')
    # columns_list=['model','tps_'+str(200),'fps_'+str(200)]
    # tps_fps_df=pd.DataFrame(np.column_stack([model_list,tps,fps]),columns=columns_list)
    # tps_fps_df.describe()
    confusion = tf.confusion_matrix(labels=tf.convert_to_tensor(eval_labels), predictions=tf.convert_to_tensor(preds), num_classes=2)

    with tf.Session():
      confusion = tf.Tensor.eval(confusion)
      print("Confusion: ",confusion)
      tn = confusion[0][0]
      fp = confusion[0][1]
      fn = confusion[1][0]
      tp = confusion[1][1]
      tpr = tp/(tp+fn)
      fpr = fp/(tn+fp)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      num_epochs=1,
      shuffle=False)

    train_results = nodet_classifier.evaluate(input_fn=eval_input_fn)
    print(train_results)
    print("No of positive examples in training is ",positive)
    print("No of negative examples in training is ",negative)

    output_str= "\n"+output_str+"train acc: "+str(train_results['accuracy'])+" test acc: "+str(test_results['accuracy'])+"\n"
    output_str = output_str+"TPR: "+str(round(tpr,4))+" "+"FPR: "+str(round(fpr,4))+" for epochs: "+str(test_results['global_step'])+"\n\n"

  print(output_str)
  tps_fps_df.to_csv('../../../tps&fps/'+model_list[0]+'.csv')

# Run the TF App once the main function is called
if __name__ == "__main__":
  tf.app.run()