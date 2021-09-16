# coding=utf-8
import tensorflow as tf
import numpy as np
import argparse
import dataprocessor as dp
import ConfMapper as cm
from pathlib import Path
from datetime import datetime
import pickle
import re
import os.path


# Command Line Parameters
parser = argparse.ArgumentParser(description="default configuration: "
                                             "batch_size = 60\n"
                                             "c_length = 19\n"
                                             "num_train = 11470\n"
                                             "embedding_size = 128\n"
                                             "epoches = int(num_train / batch_size)\n"
                                             "num_training_steps = epoches * 500")
parser.add_argument('-b', '--batch_size', type=int, default=60)
parser.add_argument('-nn', '--name_num', default='01')
parser.add_argument('-bn', '--base_num', type=int, default=1)
parser.add_argument('-en', '--epoch_num', type=int, default=1200)
parser.add_argument("-es", "--embedding_size", type=int,default=64)
parser.add_argument('-use_GPU', type=str, default=None)
args = parser.parse_args()
use_GPU = args.use_GPU
if use_GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = use_GPU
batch_size = args.batch_size
name_num = args.name_num
base_num = args.base_num
embedding_size = args.embedding_size
epoc = args.epoch_num
c_length = 19
num_train = 11470
keep_rate = 0.5
epoches = int(num_train / batch_size)
num_training_steps = epoches * epoc

# data restoring directory
directory0 = "data/"
directory1 = "data_baseline_CNN/"
accu_file_name = 'baselineCNN_epoch%d_basenum%d_name%s.txt' % (epoc, base_num,name_num)
data_name = directory0 + directory1 + accu_file_name
print("Accuracy data will be restored in %s" % data_name)
# CNN model


def embedding_layer(input_seq, vector_size=64):
    if len(input_seq.shape) != 2:
        raise ValueError("Input array should be matrix, input shape is:", input_seq.shape)

    with tf.variable_scope("Embeddings"):
        em_table = get_weights(shape=[2, vector_size], name="embedding_table")
        return tf.nn.embedding_lookup(params=em_table, ids=input_seq)


def cnn_module(input,
               input_shape: list,
               top_layer_shape=512,
               name="CNN_module"):
    input = tf.reshape(input, input_shape)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        num_channel = 64
        kernel1 = get_weights([5, 5, 1, num_channel], "conv1_kernel")
        bias1 = get_bias([num_channel], "conv1_bias")
        kernel2 = get_weights([3, 3, num_channel, num_channel], "conv2_kernel")
        bias2 = get_bias([num_channel], "conv2_bias")
        strides = [1, 1, 1, 1]

        conv1 = tf.nn.relu(tf.nn.conv2d(input, kernel1, strides, padding="VALID") + bias1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, kernel2, strides, padding="VALID") + bias2)

        n_flat = (input_shape[-3] - 5 - 3 + 2) * (input_shape[-2] - 5 - 3 + 2) * num_channel
        conv2_flat = tf.reshape(conv2, [-1, n_flat])
        dense1 = get_dense(conv2_flat, 1024, name="dense1")
        dense2 = get_dense(dense1, top_layer_shape, name="dense2")
        return dense2


def cnn_output(input, output_shape, is_training: tf.placeholder):
    with tf.variable_scope("cnn_output", reuse=tf.AUTO_REUSE):
        output = tf.cond(is_training,
                         true_fn=lambda: get_dense(tf.nn.dropout(input, keep_prob=keep_rate),
                                                   output_shape[0] * output_shape[1],
                                                   name="dense_output", activation=None),
                         false_fn=lambda: get_dense(input, output_shape[0] * output_shape[1],
                                                    name="dense_output", activation=None))
        # output = tf.tanh(output / 40.0) * 40.0
        output = tf.nn.softmax(tf.reshape(output, [-1, output_shape[0], output_shape[1]]),
                               axis=-1)
    return output


def CNN_model(x:tf.placeholder,
              y_:tf.placeholder,
              is_training:tf.placeholder,
              basenum,
              em_size=embedding_size,
              seq_num=19):
    embeddings = embedding_layer(x, vector_size=em_size)
    shape0 = [-1, seq_num, em_size, 1]
    cnn = cnn_module(embeddings, shape0)
    y = cnn_output(cnn, cm.get_outputshape(basenum), is_training=is_training)
    y_ = tf.one_hot(y_, cm.get_outputshape(basenum)[1])
    loss = tf.reduce_mean(tf.reduce_sum(- y_ * tf.log(y + 1e-8), axis=[-2, -1]))
    return y, loss


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


# Functions
def get_weights(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.initializers.truncated_normal(stddev=0.1))


def get_bias(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.initializers.constant(0.1))


def get_dense(input, units, activation=tf.nn.relu, name="dense"):
    return tf.layers.dense(input, units, activation=activation,
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                           bias_initializer=tf.initializers.constant(0.1),
                           reuse=tf.AUTO_REUSE,
                           name=name)


# run Session
if __name__ == "__main__":
    # data pipeline
    # original output data takes values:{-1, 0, 1}
    # --> We map the data to a base 3 number taking values {0, 1, 2}
    with open('dataset/HP19trainset11470.txt', 'rb') as f:
        trainset = pickle.load(f)
    input_train = np.array(trainset['input'], dtype=np.int32)  # [1,0,0,1,1,0,0,0....]
    output_train = np.array(trainset['output'], dtype=np.int32) + 1  # [1,0,-1] |-->  [0,1,2]
    with open('dataset/HP19testset2000.txt', 'rb') as f:
        testset = pickle.load(f)
    input_test = np.array(testset['input'], dtype=np.int32)  # [1,0,0,1,1,0,0,0....]
    output_test = np.array(testset['output'], dtype=np.int32) + 1  # [1,0,-1] |-->  [0,1,2]

    # input & label placeholder:
    input_shape, output_shape = input_train.shape[1], output_train.shape[1]
    x = tf.placeholder(tf.int32, shape=[None, input_shape])
    y_ = tf.placeholder(tf.int32, shape=[None, cm.get_outputshape(base_num)[0]])
    is_training = tf.placeholder(tf.bool)

    # Model & training_step
    WDecay = 0.004 * tf.nn.l2_loss(tf.global_variables())
    y, loss = CNN_model(x, y_, is_training, basenum=base_num)
    loss = loss + WDecay
    accuracy_distribution = accuracy_metrics(y, y_, base_num=base_num)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    session = tf.Session()
    with session as sess:
        sess.run(tf.global_variables_initializer())
        idx = np.arange(num_train)
        accu_test_list = []
        for i in range(num_training_steps):
            # Training Process
            np.random.shuffle(idx)
            train_dict = {x: input_train[idx[0:batch_size]],
                          y_: cm.convert_tertodec(output_train[idx[0:batch_size]], base_num),
                          is_training: True}
            sess.run(train_step, feed_dict=train_dict)

            # Evaluation Metrics
            if i % 100 == 0:
                # train accuracy of 1000 training samples
                accu_train_dict = {x: input_train[idx[0:1000]],
                                   y_: cm.convert_tertodec(output_train[idx[0:1000]], base_num),
                                   is_training: False}
                accuracy_list_train, loss_eval = sess.run((accuracy_distribution, loss),
                                                          feed_dict=accu_train_dict)
                print("Step %d, Train Accuracy Distribution:\n" % i, accuracy_list_train)
                print("Train Cross Entropy:", loss_eval)

                # test accuracy of test sets
                accu_test_dict = {x: input_test,
                                  y_: cm.convert_tertodec(output_test, base_num),
                                  is_training: False}
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









