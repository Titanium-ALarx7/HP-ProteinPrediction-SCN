# coding=utf-8
# Copyright 2021
import tensorflow as tf
import numpy as np
import pickle
import re
import os.path
import argparse
import ConfMapper as cm
from pathlib import Path
from datetime import datetime


# Config
# Creat a parser instance as interface for command arguments
parser = argparse.ArgumentParser\
    (description="default configuration: "
     "batch_size = 60\n"
     "c_length = 19\n"
     "num_train = 11470\n"
     "embedding_size = 128\n"
     "num_attention_heads= 4\n"
     "base_num=1\n"
     "nblock = 3\n"
     "epoches = int(num_train / batch_size)\n"
     "num_training_steps = epoches * 500\n"
     "use_GPU: default is None, which means tensorflow has access to all the device,"
     "use_GPU parameter assigns values to environment variables CUDA_VISIBLE_DEVICES, "
     "to give permission to specified GPUs.")


parser.add_argument('-b','--batch_size', type=int, default=60)
parser.add_argument('--nblock', type=int, default=3)
parser.add_argument('--embeddingsize', type=int, default=128)
parser.add_argument('--nheads', type=int, default=8)
parser.add_argument('-in', '--iter_num', type=int, default=5)
parser.add_argument('-nn', '--name_num', default='01')
parser.add_argument('-bn', '--base_num', type=int, default=1)
parser.add_argument('-en', '--epoch_num', type=int, default=1200)
parser.add_argument('-use_GPU', type=str, default=None)
args = parser.parse_args()
use_GPU = args.use_GPU
if use_GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = use_GPU
batch_size = args.batch_size
nblock = args.nblock
embedding_size = args.embeddingsize
num_attention_heads = args.nheads
iter_num = args.iter_num
name_num = args.name_num
base_num = args.base_num
epoc = args.epoch_num
c_length = 19
num_train = 11470
epoches = int(num_train / batch_size)
num_training_steps = epoches * epoc
iter_width = 12
# data restoring directory
directory0 = "data/"
directory1 = 'data_HPSCC/'
accu_file_name = 'HPSCC_epoch%d_basenum%d_name%s.txt' % (epoc, base_num,name_num)
data_name = directory0 + directory1 + accu_file_name


# func0
def create_initializer(initializer_range=0.02):
   """Creates a `truncated_normal_initializer` with the given range."""
   return tf.truncated_normal_initializer(stddev=initializer_range)


# func1
def flatten_tensor_to_2d(tensor):
    if tensor.shape.dim <= 2:
        return tensor
    else:
        width = tensor.shape[-1]
        output = tf.reshape(tensor, [-1, width])
    return output


# func2
def get_shape_list(tensor):
    # 获取动态shape（class:Tensor）的list
    shape = tensor.shape.as_list()
    # placeholder的None维度无法获取
    non_static_index = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_index.append(index)
    if not non_static_index:
        return shape
    dyn_shape = tf.shape(tensor)
    for index in non_static_index:
        shape[index] = dyn_shape[index]
    return shape


# func3
def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


# func4
def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


# layer 0 * 2
def HP_layer_lookup(input_seqs,
                    n_hidden=iter_width,
                    n_neuron=50,
                    initializer_range=0.02):
    # 检查input dtype&shape
    if input_seqs.shape.ndims != 2:
        raise ValueError("input_seqs has unmatched ndims: %d" % input_seqs.shape.ndims)
    # inputseqs就是x, 是一个int32，[-1,19]的tensor，由(0,1)两种取值构成
    with tf.variable_scope("HP_layer", reuse=tf.AUTO_REUSE):
        W_emb1 = tf.get_variable("W_emb1",
                                 [2, n_hidden * n_neuron],
                                 dtype=tf.float32,
                                 initializer=create_initializer(initializer_range))
        W_emb2 = tf.get_variable("W_emb2",
                                 [2, n_hidden * n_neuron],
                                 dtype=tf.float32,
                                 initializer=create_initializer(initializer_range))
        B_emb1 = tf.get_variable("B_emb1",
                                 [2, n_neuron],
                                 initializer=tf.constant_initializer(0.1))
        B_emb2 = tf.get_variable("B_emb2",
                                 [2, n_hidden],
                                 initializer=tf.constant_initializer(0.1))
        WHP1 = tf.reshape(tf.nn.embedding_lookup(W_emb1, input_seqs), [-1, c_length, n_hidden, n_neuron])
        bHP1 = tf.reshape(tf.nn.embedding_lookup(B_emb1, input_seqs), [-1, c_length, n_neuron])
        WHP2 = tf.reshape(tf.nn.embedding_lookup(W_emb2, input_seqs), [-1, c_length, n_neuron, n_hidden])
        bHP2 = tf.reshape(tf.nn.embedding_lookup(B_emb2, input_seqs), [-1, c_length, n_hidden])
    return (WHP1, bHP1,WHP2, bHP2)


def HP_layer(iter_xx, WHP1, bHP1,WHP2, bHP2,
             n_hidden=iter_width,
             n_neuron=50):
    with tf.variable_scope("HP_layer", reuse=tf.AUTO_REUSE):
        iter_xx = tf.reshape(iter_xx, [-1, c_length, n_hidden, 1])
        HP_layer1 = gelu(tf.reduce_sum(iter_xx * WHP1, axis=-2) + bHP1)
        HP_layer1 = tf.reshape(HP_layer1, [-1, c_length, n_neuron, 1])
        HP_layer2 = gelu(tf.reduce_sum(HP_layer1 * WHP2, axis=-2) + bHP2)
    return HP_layer2


# layer 1
def attention_layer(input_tensor,
                    num_attention_heads=num_attention_heads,
                    size_per_head=int(embedding_size / num_attention_heads),
                    attention_keep_probs=0.9,
                    initializer_range=0.02,
                    act=None,
                    name="attention_layer"):
    def transpose_to_multiheads(tensor):
        # id_nums = tensor.shape[1]
        tensor = tf.reshape(tensor, [-1,id_nums, num_attention_heads, size_per_head])
        output_tensor = tf.transpose(tensor, [0, 2, 1, 3])
        return output_tensor

    if input_tensor.shape.ndims != 3:
        print(input_tensor.shape)
        raise ValueError("One batch of embedding 19mer should be rank 3")
    # if size_per_head * num_attention_heads != input_tensor.shape[-1]:
    # raise ValueError("size_per_head * num_attention_heads == hidden_size")

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = input_tensor.shape.as_list()
        id_nums = shape[1]
        query_layer = tf.layers.dense(input_tensor,
                                      num_attention_heads * size_per_head,
                                      activation=act,
                                      name="query",
                                      kernel_initializer=create_initializer(initializer_range),
                                      reuse=tf.AUTO_REUSE)
        key_layer = tf.layers.dense(input_tensor,
                                    num_attention_heads * size_per_head,
                                    activation=act,
                                    name="key",
                                    kernel_initializer=create_initializer(initializer_range),
                                    reuse=tf.AUTO_REUSE)
        value_layer = tf.layers.dense(input_tensor,
                                      num_attention_heads * size_per_head,
                                      activation=act,
                                      name="value",
                                      kernel_initializer=create_initializer(initializer_range),
                                      reuse=tf.AUTO_REUSE)
        q = transpose_to_multiheads(query_layer)  # [-1, num_heads, id_num, head_size]
        k = transpose_to_multiheads(key_layer)
        v = transpose_to_multiheads(value_layer)
        attention_score = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2]), name="attention_score")
        a_s_probs = tf.nn.softmax(tf.multiply(attention_score, 1.0 / tf.sqrt(float(size_per_head))),
                                  axis=-1,
                                  name="attention_score_probs")
        a_s_probs = tf.nn.dropout(a_s_probs, attention_keep_probs, name="attention_score_dropout")
        context_layer = tf.transpose(tf.matmul(a_s_probs, v), [0, 2, 1, 3])
        context_layer = tf.reshape(context_layer,
                                   [-1, id_nums, num_attention_heads * size_per_head],
                                   name="context_layer")

    return context_layer


# module 0
def encoder_block(input_tensor,
                  iter_width=iter_width,
                  dense_width=embedding_size,
                  initializer_range=0.02,
                  intermediate_act=gelu,
                  intermediate_size=768,
                  blockname="encoder_block",
                  dense_keep_prob=0.9,
                  attention_keep_probs=0.9):
    with tf.variable_scope(blockname, reuse=tf.AUTO_REUSE):
        attention_output = attention_layer(input_tensor, attention_keep_probs=attention_keep_probs)
        dense1 = tf.layers.dense(attention_output,
                                 iter_width,
                                 kernel_initializer=create_initializer(initializer_range),
                                 activation=None,
                                 name="Dense1",
                                 reuse=tf.AUTO_REUSE)
        dropout_dense1 = tf.nn.dropout(dense1, keep_prob=dense_keep_prob)
        residual_norm_output1 = layer_norm(dropout_dense1 + input_tensor)
        # 等会试试在intermediate层把它们混在一起
        residual_norm_output1_ = tf.reshape(residual_norm_output1, [-1, c_length, iter_width, 1])
        W_conv1 = tf.get_variable(name="W_conv1",
                                  shape=[5, 5, 1, 64],
                                  dtype=tf.float32,
                                  initializer=create_initializer(initializer_range))
        W_conv2 = tf.get_variable(name="W_conv2",
                                  shape=[4, 3, 64, 64],
                                  dtype=tf.float32,
                                  initializer=create_initializer(initializer_range))
        b_conv1 = tf.get_variable(name="b_conv1",
                                  shape=[64],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        b_conv2 = tf.get_variable(name="b_conv1",
                                  shape=[64],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.relu(tf.nn.conv2d(residual_norm_output1_, W_conv1, strides=[1, 3, 1, 1],
                                        padding='SAME') + b_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1],
                                        padding='VALID') + b_conv2)
        conv2 = tf.reshape(conv2, [-1, 4 * 64 * 10])
        dense2 = tf.layers.dense(conv2,
                                 iter_width * c_length,
                                 name="Dense2",
                                 kernel_initializer=create_initializer(initializer_range),
                                 reuse=tf.AUTO_REUSE)
        dense2_ = tf.reshape(dense2, [-1, c_length, iter_width])
        dropout_dense2 = tf.nn.dropout(dense2_, keep_prob=dense_keep_prob)
        residual_norm_output2 = layer_norm(dropout_dense2 + residual_norm_output1)
        residual_norm_output2_ = tf.reshape(residual_norm_output2, [-1, c_length, iter_width, 1])
        W_conv3 = tf.get_variable(name="W_conv3",
                                  shape=[5, 5, 1, 64],
                                  dtype=tf.float32,
                                  initializer=create_initializer(initializer_range))
        W_conv4 = tf.get_variable(name="W_conv4",
                                  shape=[3, 3, 64, 64],
                                  dtype=tf.float32,
                                  initializer=create_initializer(initializer_range))
        b_conv3 = tf.get_variable(name="b_conv3",
                                  shape=[64],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        b_conv4 = tf.get_variable(name="b_conv4",
                                  shape=[64],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        conv3 = tf.nn.relu(tf.nn.conv2d(residual_norm_output2_, W_conv3, strides=[1, 1, 1, 1],
                                        padding='VALID') + b_conv3)
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W_conv4, strides=[1, 1, 1, 1],
                                        padding='VALID') + b_conv4)
        conv4 = tf.reshape(conv4, [-1, 13 * 64 * (iter_width - 6)])
        dense3 = tf.layers.dense(conv4,
                                 iter_width * c_length,
                                 name="Dense3",
                                 kernel_initializer=create_initializer(initializer_range),
                                 reuse=tf.AUTO_REUSE)
        dense3 = tf.reshape(dense3,[-1, c_length, iter_width])
        dropout_dense3 = tf.nn.dropout(dense3, keep_prob=dense_keep_prob)
        residual_norm_output3 = layer_norm(dropout_dense3 + residual_norm_output2)
    return residual_norm_output3


# model
def attentionCNN_model(input_tensor,
                      num_of_blocks=nblock,
                      dense_width=embedding_size,
                      iter_width=iter_width,
                      initializer_range=0.02,
                      intermediate_act=tf.nn.relu,
                      intermediate_size=768,
                      dense_keep_prob=0.9,
                      attention_keep_probs=0.9,
                      return_all_layer=False,
                      encoder_block_name="encoder_block"):
    all_block_outputs = []
    prev_block_output = input_tensor
    for layer_index in range(num_of_blocks):
            layer_output = encoder_block(prev_block_output,
                                         iter_width=iter_width,
                                         dense_width=dense_width,
                                         initializer_range=initializer_range,
                                         intermediate_act=intermediate_act,
                                         intermediate_size=intermediate_size,
                                         dense_keep_prob=dense_keep_prob,
                                         attention_keep_probs=attention_keep_probs,
                                         blockname="%s%d"
                                         % (encoder_block_name, layer_index))
            all_block_outputs.append(layer_output)
            prev_block_output = layer_output
    if return_all_layer:
        return all_block_outputs
    else:
        return all_block_outputs[-1]


def output_layer(input, base_num=base_num):
    with tf.name_scope("output_layer"):
        dense_output = tf.layers.dense(tf.reshape(input, [-1, c_length * iter_width]),
                                       1024,
                                       kernel_initializer=create_initializer(),
                                       activation=tf.nn.relu,
                                       name="dense_output")
        output = tf.layers.dense(dense_output,
                                 cm.get_outputshape(base_num)[0] * cm.get_outputshape(base_num)[1],
                                 kernel_initializer=create_initializer(),
                                 name="y_output")
        output_probs = tf.nn.softmax(tf.reshape(output,
                                [-1, cm.get_outputshape(base_num)[0], cm.get_outputshape(base_num)[1]]),
                                axis=-1, name="output_probs")
        return output_probs


# Training Model
def ANN_HPSCC_model(x, y_, iter_xx, base_num=base_num):
    iteration_xx = iter_xx
    WHP1, bHP1, WHP2, bHP2 = HP_layer_lookup(x)
    iteration_xx_output = None
    for i in range(iter_num):
        iter_output = HP_layer(iteration_xx, WHP1, bHP1, WHP2, bHP2)
        iteration_xx_output = attentionCNN_model(iter_output,
                                                 dense_keep_prob=dropout_keep_probs,
                                                 attention_keep_probs=dropout_keep_probs)
        iteration_xx = tf.nn.softmax(iteration_xx_output, axis=-1)
    y = output_layer(iteration_xx_output)
    y_ = tf.one_hot(y_, cm.get_outputshape(base_num)[1])
    with tf.name_scope('loss_optimization'):
        cross_entropy = tf.reduce_mean(tf.reduce_sum(-tf.log(y + 1e-8) * y_,
                                       reduction_indices=[-2, -1]), name="cross_entropy")
    return y, cross_entropy


# Evaluation Metrics
def accuracy_metrics(y_pred, y_label, base_num, num_seq=19):
    with tf.name_scope('Accuracy_metrics'):
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        y_label = tf.cast(y_label, tf.int32)
        y_pred = cm.convert_dectoter(y_pred, base_num)
        y_label = cm.convert_dectoter(y_label, base_num)
        accu_per_position = tf.equal(y_pred, y_label)  # shape: [Batch, 19]
        accu_per_seq = tf.reduce_sum(tf.cast(accu_per_position, tf.int32), axis=-1)  # shape: [Batch]
        accs_train_ones = tf.ones_like(accu_per_seq)
        accuracy_distribution = []
        accuracy_distribution_test = []
        for i in range(num_seq + 1):
            accuracy_i = tf.equal(accu_per_seq, accs_train_ones * i)  # 也就是Batch 个 i
            accuracy_distribution.append(tf.reduce_mean(tf.cast(accuracy_i, tf.float32)))
            accuracy_distribution_test.append(tf.reduce_mean(tf.cast(accuracy_i, tf.float32)))
    return accuracy_distribution


# feed_dict
with tf.name_scope('input_output_placeholder'):
    x = tf.placeholder(tf.int32, [None, c_length], 'input_seq_x')
    iter_xx = tf.placeholder(tf.float32, [None, c_length, iter_width], 'iter_xx')
    y_ = tf.placeholder(tf.int32, [None, cm.get_outputshape(base_num)[0]], 'output_label_y')
    dropout_keep_probs = tf.placeholder(tf.float32, [], 'dropout_keep_probs')


y, loss = ANN_HPSCC_model(x, y_, iter_xx)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
accuracy_distribution = accuracy_metrics(y, y_, base_num=base_num)


def main_func(input_train, output_train, input_test, output_test, base_num=base_num):
    with tf.Session() as sess:
        # Session initialization
        sess.run(tf.global_variables_initializer())
        idx = np.arange(num_train)
        accu_test_list = []
        iter_xx_init = np.ones((batch_size, c_length, iter_width), np.float32) * (1.0 / iter_width)
        for i in range(num_training_steps):
            # Training Process
            np.random.shuffle(idx)
            train_dict = {x: input_train[idx[0:batch_size]],
                          y_: cm.convert_tertodec(output_train[idx[0:batch_size]], base_num),
                          iter_xx: iter_xx_init,
                          dropout_keep_probs: 0.9}
            sess.run(train_step, feed_dict=train_dict)

            # Evaluation Metrics
            if i % 100 == 0:
                # train accuracy of 1000 training samples
                accu_train_dict = {x: input_train[idx[0:1000]],
                                   y_: cm.convert_tertodec(output_train[idx[0:1000]], base_num),
                                   iter_xx: np.ones((1000, c_length, iter_width), np.float32) * (1.0 / iter_width),
                                   dropout_keep_probs: 1.0}
                accuracy_list_train, loss_eval = sess.run((accuracy_distribution, loss),
                                                          feed_dict=accu_train_dict)
                print("Step %d, Train Accuracy Distribution:\n" % i, accuracy_list_train)
                print("Train Cross Entropy:", loss_eval)

                # test accuracy of test sets
                accu_test_dict = {x: input_test,
                                  y_: cm.convert_tertodec(output_test, base_num),
                                  iter_xx: np.ones((2000, c_length, iter_width), np.float32) * (1.0 / iter_width),
                                  dropout_keep_probs: 1.0}
                accuracy_list_test = sess.run(accuracy_distribution, feed_dict=accu_test_dict)
                accu_test_list.append(accuracy_list_test[-1])

                with open(data_name, 'wb') as data:
                    pickle.dump(accu_test_list, data)

                print("Step %d, Test Accuracy Distribution:\n" % i, accuracy_list_test)
                if len(accu_test_list) > 50:
                    print("*" * 50, "Test set latest accuracies:", sep="\n")
                    print(accu_test_list[-50:-1], end="\n" + "*" * 50 + '\n')
                else:
                    print("*" * 50, "Test set latest accuracies:", sep="\n")
                    print(accu_test_list, end="\n" + "*" * 50 + '\n')


if __name__ == "__main__":
    # Data Pipeline:
    with open('dataset/HP19trainset11470.txt', 'rb') as f:
        trainset = pickle.load(f)
    input_train = np.array(trainset['input'], dtype=np.int32)  # [1,0,0,1,1,0,0,0....]
    output_train = np.array(trainset['output'], dtype=np.int32) + 1  # [1,0,-1] |-->  [0,1,2]
    with open('dataset/HP19testset2000.txt', 'rb') as f:
        testset = pickle.load(f)
    input_test = np.array(testset['input'], dtype=np.int32)  # [1,0,0,1,1,0,0,0....]
    output_test = np.array(testset['output'], dtype=np.int32) + 1  # [1,0,-1] |-->  [0,1,2]

    main_func(input_train, output_train, input_test, output_test, base_num=base_num)


