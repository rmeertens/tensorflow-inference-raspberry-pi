checkpoint_name = 'mobilenet_v2_0.35_96' #@param

# setup path
import sys
sys.path.append('/home/pi/models/research/slim')

import tensorflow as tf
# from nets.mobilenet import mobilenet_v2

# from datasets import imagenet
import PIL

import numpy as np
img = np.array(PIL.Image.open('panda.jpg').resize((96,96))).astype(np.float) / 128 - 1
gd = tf.GraphDef.FromString(open(checkpoint_name + '_frozen.pb', 'rb').read())
inp, predictions = tf.import_graph_def(gd,  return_elements = ['input:0', 'MobilenetV2/Predictions/Reshape_1:0'])

with tf.Session(graph=inp.graph):
  x = predictions.eval(feed_dict={inp: img.reshape(1, 96,96, 3)})

# label_map = imagenet.create_readable_names_for_imagenet_labels()
print(x.argmax())
# print("Top 1 Prediction: ", x.argmax(),label_map[x.argmax()], x.max())
