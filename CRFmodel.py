# coding=utf-8
import tensorflow as tf
import numpy as np
import argparse
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
                                             "num_training_steps = epoches * 500\n"
                                             "use_GPU: use all GPUs as default; use str 1,2,3 to assign specified usage. ")
parser.add_argument('-b', '--batch_size', type=int, default=60)
parser.add_argument('-nn', '--name_num', default='01')
parser.add_argument('-en', '--epoch_num', type=int, default=1200)
parser.add_argument('-psi_size', type=int, default=4)
parser.add_argument('-use_GPU', type=str, default=None)
parser.add_argument("-ulp", type=float, default=0.8)
args = parser.parse_args()

use_GPU = args.use_GPU
if use_GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = use_GPU
batch_size = args.batch_size
name_num = args.name_num
epoc = args.epoch_num
potential_field_size = args.psi_size
use_label_probs = args.ulp
c_length = 19
num_train = 11470
keep_rate = 0.5
epoches = int(num_train / batch_size)
num_training_steps = epoches * epoc

# data restoring directory
directory0 = "data/"
directory1 = "data_CRF_CNN/"
accu_file_name = 'CRF_epoch%d_psiSize%d_label-probs%d_name%s.txt' % (epoc, potential_field_size, use_label_probs, name_num)
data_name = directory0 + directory1 + accu_file_name
print("Accuracy data will be restored in %s" % data_name)

# CNN model
def embedding_layer(input_seq, vector_size=128):
    if len(input_seq.shape) != 2:
        raise ValueError("Input array should be matrix, input shape is:", input_seq.shape)

    with tf.variable_scope("Embeddings", reuse=tf.AUTO_REUSE):
        em_table = get_weights(shape=[2, vector_size], name="embedding_table")
        return tf.nn.embedding_lookup(params=em_table, ids=input_seq)


def cnn_module(input,
               input_shape: list,
               top_layer_shape=64 * 19,
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


def CRF_output(input, y_label, use_label, is_training: tf.placeholder, potential_field_size=4, iter_num=3):  # requested input tensor shape: [batch, 19, hidden]
    if potential_field_size < 2:
        raise ValueError("PSI potential field should be larger than two residues.")
    y_label = tf.one_hot(y_label, depth=3)
    ps = potential_field_size

    output_shape = [19, 64]
    output = tf.cond(is_training,
                     true_fn=lambda: get_dense(tf.nn.dropout(input, keep_prob=keep_rate),
                                               output_shape[0] * output_shape[1],
                                               name="dense_output", activation=None),
                     false_fn=lambda: get_dense(input, output_shape[0] * output_shape[1],
                                                name="dense_output", activation=None))
    output = tf.reshape(output, [-1, output_shape[0], output_shape[1]])
    Phi = get_dense(output, units=3, activation=None)
    y_probs = tf.nn.softmax(Phi, axis=-1)

    Psi_shape = np.ones(ps + 1) * 3
    Psi_shape[0] = 1
    Psi = get_weights(name="Psi_matrix", shape=Psi_shape)

    def use_y_probs(y_probs=y_probs):
        for k in range(iter_num):
            y_paddings = tf.concat([tf.zeros_like(y_probs)[:, 0:ps - 1, :],
                                    y_probs,
                                    tf.zeros_like(y_probs)[:, 0:ps - 1, :]], axis=-2)
            y_aux_list = []
            for i in range(2 * (ps - 1)):  # Move Order: left 3, left 2, left 1, right 1, right2, right 3....
                y_aux_list.append(y_paddings[:, i:i + 19, :])
            y_psi = 0
            index0 = - np.arange(1, ps + 1, dtype=np.int32)
            index1 = np.eye(ps, dtype=np.int32) * 2 + np.ones(ps)
            index1 = np.concatenate((-np.ones([ps, 1]), 19 * np.ones([ps, 1]), index1), axis=-1)
            for i in range(ps):
                shape_index = np.delete(index1, -i, axis=0)
                y_psi = Psi
                for j in range(ps - 1):
                    y_psi *= tf.reshape(y_aux_list[i + j], shape_index[j])
                y_psi = tf.reduce_sum(y_psi, axis=np.delete(index0, i))
            y_probs = tf.nn.softmax(y_psi + Phi, axis=-1)
        return y_probs

    def use_y_labels():
        y_paddings = tf.concat([tf.zeros_like(y_label)[:, 0:ps - 1, :],
                                y_label,
                                tf.zeros_like(y_label)[:, 0:ps - 1, :]], axis=-2)
        y_aux_list = []
        for i in range(2 * (ps - 1)):  # Move Order: left 3, left 2, left 1, right 1, right2, right 3....
            y_aux_list.append(y_paddings[:, i:i + 19, :])
        y_psi = 0
        index0 = - np.arange(1, ps + 1, dtype=np.int32)
        index1 = np.eye(ps, dtype=np.int32) * 2 + np.ones(ps)
        index1 = np.concatenate((-np.ones([ps, 1]), 19 * np.ones([ps, 1]), index1), axis=-1)
        for i in range(ps):
            shape_index = np.delete(index1, -i, axis=0)
            y_psi = Psi
            for j in range(ps - 1):
                y_psi *= tf.reshape(y_aux_list[i + j], shape_index[j])
            y_psi = tf.reduce_sum(y_psi, axis=np.delete(index0, i))
        y_probs = tf.nn.softmax(y_psi + Phi, axis=-1)
        return y_probs

    y = tf.cond(use_label, true_fn=use_y_labels, false_fn=use_y_probs)
    loss = tf.reduce_mean(tf.reduce_sum(- y_label * tf.log(y + 1e-8), axis=[-2, -1]))
    return y, loss


def CRF_CNN_model(x: tf.placeholder,
                  y_: tf.placeholder,
                  is_training: tf.placeholder,
                  use_label: tf.placeholder,
                  potential_field_size=4,
                  probs_iter=3,
                  seq_num=19):
    em_size = 128
    embeddings = embedding_layer(x)
    shape0 = [-1, seq_num, em_size, 1]
    cnn = cnn_module(embeddings, shape0)
    return CRF_output(cnn, y_, use_label,
                      is_training=is_training,
                      potential_field_size=potential_field_size,
                      iter_num=probs_iter)


# Evaluation Metrics
def CRF_accuracy_metrics(y_pred, y_label, num_seq=19):
    with tf.name_scope('Accuracy_metrics'):
        y_label = tf.cast(y_label, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        accu_per_position = tf.equal(y_pred, y_label)  # shape: [Batch, 19]
        accu_per_seq = tf.reduce_sum(tf.cast(accu_per_position, tf.int32), axis=-1)  # shape: [Batch]
        accs_train_ones = tf.ones_like(accu_per_seq)
        accuracy_distribution = []
        accuracy_distribution_test = []
        for i in range(num_seq + 1):
            accuracy_i = tf.equal(accu_per_seq, accs_train_ones * i)  # namely batch num of i
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


def get_CRF_output_maker():
    base34 = np.zeros([81, 4], dtype=np.int32)
    for i in range(81):
        sum = i
        for j in range(4):
            base34[i, j] = sum % 3
            sum = sum // 3
    base34_tensor = tf.constant(base34, dtype=tf.int32)

    def CRF_prediction_output4(Phi, Psi,start_index):
        list_output = []

        for i in range(81):
            E = 0
            for j in range(4):
                if j < 3:
                    E += Phi[:, j + start_index, base34[i, j]] + Psi[base34[i, j], base34[i, j + 1]]
                else: E += Phi[:, j + start_index, base34[i, j]]
                E = tf.reshape(E, shape=[-1, 1])
            if i == 0:
                list_output = E
            else:
                list_output = tf.concat([list_output, E], axis=-1)
        max_index = tf.argmax(list_output, axis=-1)
        return tf.nn.embedding_lookup(base34_tensor, max_index)

    return CRF_prediction_output4


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
    y_ = tf.placeholder(tf.int32, shape=[None, output_shape])
    is_training = tf.placeholder(tf.bool)
    use_label = tf.placeholder(tf.bool)

    # Model & training_step
    y, loss = CRF_CNN_model(x, y_, is_training, use_label,
                            potential_field_size=potential_field_size)
    accuracy_distribution = CRF_accuracy_metrics(y, y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        idx = np.arange(num_train)
        accu_test_list = []
        for i in range(num_training_steps):
            # Training Process
            if i <= epoches * 50:
                prob_use = (np.random.uniform() <= use_label_probs)
            else:
                prob_use = (np.random.uniform() <= use_label_probs - 0.3)
            np.random.shuffle(idx)
            train_dict = {x: input_train[idx[0:batch_size]],
                          y_: output_train[idx[0:batch_size]],
                          is_training: True,
                          use_label: prob_use}
            sess.run(train_step, feed_dict=train_dict)


            # Evaluation Metrics
            if i % 100 == 0:
                # train accuracy of 1000 training samples
                accu_train_dict = {x: input_train[idx[0:1000]],
                                   y_: output_train[idx[0:1000]],
                                   is_training: False,
                                   use_label: False}
                accuracy_list_train, loss_eval = sess.run((accuracy_distribution, loss),
                                                      feed_dict=accu_train_dict)
                print("*" * 50)
                print("Step %d, Train Accuracy Distribution:\n" % i, accuracy_list_train)
                print("Train cross entropy:", loss_eval, end="\n" + "*" * 50 + "\n")

                # test accuracy of test sets
                accu_test_dict = {x: input_test,
                                  y_: output_test,
                                  is_training: False,
                                  use_label: False}
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











