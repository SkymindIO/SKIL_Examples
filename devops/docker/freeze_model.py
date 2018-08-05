import sys, os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32, name="keep_prob_input")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2) 
  return y_conv, keep_prob
 

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework import graph_io

# Import data
work_directory = 'data_directory'
saver_write_version = 2

mnist = input_data.read_data_sets(work_directory, one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name="input_node")

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10], name="input_labels_node")

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

softmax = tf.nn.softmax(y_conv, name="output_node")

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

checkpoint_prefix = os.path.join(work_directory, "saved_checkpoint")
checkpoint_meta_graph_file = os.path.join(work_directory,
                                          "saved_checkpoint.meta")
checkpoint_state_name = "checkpoint_state"
input_graph_name = "input_graph.pb"
output_graph_name = "output_graph.pb"

print("\nTraining model...")

# We'll later add these model accuraries to the uploaded SKIL model evaluation, manually.
train_accuracy = 0
test_accuracy = 0

# Later, we will use the two arrays below to calculate the model evaluation through the feedback endpoint
test_guesses = [] # This will contain the predicted labels array
test_correct = [] # This will contain the actual labels array

steps = 1001
test_images = mnist.test.images
test_labels = mnist.test.labels

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(steps):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('\nTesting model...')    
  # These two string arrays will be used for the feedback endpoint at the end of the notebook
  test_guesses = tf.argmax(y_conv, 1).eval(feed_dict={x: test_images, keep_prob: 1.0}).astype(str)
  test_correct = tf.argmax(y_, 1).eval(feed_dict={y_: test_labels}).astype(str)
  
  test_accuracy = accuracy.eval(feed_dict={
      x: test_images, y_: test_labels, keep_prob: 1.0})
  print('test accuracy %g' % test_accuracy)
  
  print("\nSaving checkpoint...")

  saver = saver_lib.Saver(write_version=saver_write_version)
  checkpoint_path = saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  graph_io.write_graph(sess.graph, work_directory, input_graph_name)

  input_graph_path = os.path.join(work_directory, input_graph_name)
  input_saver_def_path = ""
  input_binary = False
  output_node_names = "output_node"
  restore_op_name = "save/restore_all"
  filename_tensor_name = "save/Const:0"
  output_graph_path = os.path.join(work_directory, output_graph_name)
  clear_devices = False
  input_meta_graph = checkpoint_meta_graph_file

  print("\nFreezing graph...")
    
  freeze_graph.freeze_graph(
        input_graph_path,
        input_saver_def_path,
        input_binary,
        checkpoint_path,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph_path,
        clear_devices,
        "",
        "",
        input_meta_graph,
        checkpoint_version=saver_write_version)
  print("\nGraph frozen successfully!")
