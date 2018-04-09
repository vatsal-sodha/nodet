import tensorflow as tf
from google.protobuf.json_format import MessageToJson

file = "../../../../models/research/object_detection/data/test.record"
fileNum=1
for example in tf.python_io.tf_record_iterator(file):
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))
    # with open("RESULTS/image_{}".format(fileNum),"w") as text_file:
        # print(jsonMessage,file=text_file)
    print(jsonMessage)
    fileNum+=1