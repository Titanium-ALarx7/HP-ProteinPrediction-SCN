import numpy as np
import copy as cp
import tensorflow as tf
import pickle
import time


def get_outputshape(base_num, num_seq=19):
    return [num_seq // base_num + (num_seq % base_num != 0), np.power(3, base_num)]


def convert_tertodec(input_seq: np.array, base_num, num_seq=19) -> np.array:
    if base_num <= 0 or base_num > 19:
        raise ValueError
    i, k = 0, 0
    output = np.zeros(shape=[np.shape(input_seq)[0],
                             num_seq // base_num + (num_seq % base_num != 0)])
    while i < num_seq:
        for j in range(base_num):
            if i + base_num < num_seq:
                output[:, k] += input_seq[:, i + j] * np.power(3, j)
            else:
                output[:, k] += input_seq[:, num_seq - base_num + j] * np.power(3, j)
        k += 1
        i += base_num
    return output


def convert_dectoter(input_seq: np.array, base_num, num_seq=19) -> np.array:
    if base_num <= 0 | base_num > 19:
        raise ValueError
    input = input_seq
    output = None
    for j in range(num_seq // base_num + (num_seq % base_num != 0)):
        iter_array = tf.reshape(input[:, j], [-1, 1])
        for k in range(base_num):
            if j == 0 and k == 0:
                output = iter_array % 3
                iter_array = iter_array // 3

            elif (j + 1) * base_num < num_seq:
                output = tf.concat([output, iter_array % 3], axis=-1)
                iter_array = iter_array // 3
            else:
                if k >= ((j + 1) * base_num - num_seq):
                    output = tf.concat([output, iter_array % 3], axis=-1)
                iter_array = iter_array // 3
    return output
    # if i + base_num > num_seq: 则将最后一个十进制拆出来的数字只取最后几位


if __name__=="__main__":
    """
    output_test is a np.array test set with shape [2000, 19]
    Here, we employed a script to test ConfMapper methods work without error.
    An array which is converted by tertodec  should return identical to itself after converted by dectoter again.
    """
    with open('dataset/HP19testset2000.txt', 'rb') as f:
        testset = pickle.load(f)
    input_test = np.array(testset['input'], dtype=np.int32)  # [1,0,0,1,1,0,0,0....]
    output_test = np.array(testset['output'], dtype=np.int32) + 1  # [1,0,-1] |-->  [0,1,2]
    with tf.Session() as sess:
        for i in range(1, 10):
            a = time.clock()
            new = convert_tertodec(output_test[:2000], i)
            new = convert_dectoter(new, i).eval()
            # print(new == output_test[:10])
            print("Time cost:", time.clock() - a)
