# main function generates a partition of train set and test set(with given number of samples)
# Script output is a binary txt file containing lists of inputs and outputs.
import numpy as np
import random
import math
import sys
import os
import pickle
import re
import argparse
from os import walk
from os.path import join, isfile


# transforming strings into list data
def formalize_data(line: str) -> tuple:
    for c in ['\s','\[','\]','\(','\)']:
        line = re.sub(c, '', line)
    line = [int(x) for x in line.split(',')]
    xy_conf = np.reshape(line,(-1, 2))
    c_num = xy_conf.shape[0]
    relat_conf = np.zeros([c_num -1, 2])
    output_conf = np.zeros((c_num))
    for i in range(c_num): # o[0]=[0,0]; o[1] =
        if i>0:
            relat_conf[i-1] = xy_conf[i]-xy_conf[i-1]
        if i>1:
            dcross = relat_conf[i-1,0]*relat_conf[i-2,1]- relat_conf[i-1,1]* relat_conf[i-2,0] # 外积
            if dcross > 0: output_conf[i-1] = 1
            if dcross < 0: output_conf[i-1] = -1
    output_conf_reverse = output_conf[::-1]
    for i in range(c_num):
        if sum(abs(output_conf[0:i]))==0 and output_conf[i]!=0: sign_hp = output_conf[i]
        if sum(abs(output_conf_reverse[0:i]))==0 and output_conf_reverse[i]!=0:
            sign_hp_reverse =output_conf_reverse[i]
    output_conf = sign_hp * output_conf
    output_conf_reverse = sign_hp_reverse*output_conf_reverse
    return (output_conf, output_conf_reverse)


# Dividing whole dataset into two parts: one set with all the different sequence,
# the other with all the reversed sequence.
def wash_data(dir="dataset/"):
    with open(dir + dataname + 'mer_conf_data.txt', 'rb') as f:
        dict_data = pickle.load(f)
    # keys: ['num_of_sample', 'chain_length', 'input_HPs', 'output_confs']
    # Number of HP Seqs= num_of_sample / 2
    n = dict_data['num_of_sample']
    input = np.array(dict_data['input_HPs'])
    output = np.array(dict_data['output_confs'])
    washed_input = []
    washed_output = []
    reversed_input = []
    reversed_output = []
    for i in range(n):
        if i == 0:
            washed_input.append(input[i])
            washed_output.append(output[i])
        is_reversed = False
        for j in range(len(washed_input)):
            if (input[i] == washed_input[j][::-1]).all():
                print('Repeated i:',i,'j:' , j)
                is_reversed = True
                break
        if not is_reversed:
            washed_input.append(input[i])
            washed_output.append(output[i])
    n_washed = len(washed_input)
    print(len(washed_input))
    for j in range(n_washed):
        for i in range(n):
            if (input[i] == washed_input[j][::-1]).all():
                print('reversed appended i:', i,'j:' ,j)
                reversed_input.append(input[i])
                reversed_output.append(output[i])
                break
    n_reversed = len(reversed_input)
    symmetry_id = []
    for i in range(n_reversed):
        if (reversed_input[i] == reversed_input[i][::-1]).all():
            print(i)
            symmetry_id.append(i)

    with open(dir + dataname + 'mer_original%d.txt' % n_washed, 'wb') as f:
        pickle.dump((washed_input, washed_output), f)
    with open(dir + dataname + 'mer_reverse%d.txt' % n_reversed, 'wb') as f:
        pickle.dump((reversed_input, reversed_output), f)
    with open(dir + dataname + 'mer_symmetry_sample_ids.txt', 'wb') as f:
        pickle.dump(symmetry_id, f)


def creat_training_data(test_set_size=1000):
    with open(dir + dataname + 'mer_original6735.txt', 'rb') as f:
        ordered_input, ordered_output = pickle.load(f)
        ordered_input = np.array(ordered_input)
        ordered_output = np.array(ordered_output)
    with open(dir + dataname + 'mer_reverse6735.txt', 'rb') as f:
        reversed_input, reversed_output = pickle.load(f)
        reversed_input = np.array(reversed_input)
        reversed_output = np.array(reversed_output)
    with open(dir + dataname + 'mer_symmetry_sample_ids.txt', 'rb') as f:
        symmetry_ids = pickle.load(f)

    n = len(reversed_input)
    idx = np.arange(n, dtype=int)
    np.random.shuffle(idx)
    test_size = int(test_set_size / 2)
    Test_input = reversed_input[idx[0:test_size]]
    Test_output = reversed_output[idx[0:test_size]]
    for i in idx[0:test_size]:
        if i not in symmetry_ids:
            Test_input = np.append(Test_input,ordered_input[np.newaxis,i], axis=0)
            Test_output = np.append(Test_output,ordered_output[np.newaxis,i],axis=0)
        else:
            print('symmetry seq')
    Train_input = reversed_input[idx[test_size:n]]
    Train_input = np.concatenate((Train_input, ordered_input[idx[test_size:n]]),axis=0)
    Train_output = reversed_output[idx[test_size:n]]
    Train_output = np.concatenate((Train_output,ordered_output[idx[test_size:n]]),axis=0)
    num_test = len(Test_input)
    num_train = len(Train_input)
    print(Test_input.shape, Test_output.shape, Train_input.shape,Train_output.shape)

    dict_test = dict(zip(['input', 'output'], (np.array(Test_input), np.array(Test_output))))
    dict_train = dict(zip(['input', 'output'], (np.array(Train_input), np.array(Train_output))))
    filetest = dataname + 'testset%d.txt' % num_test
    filetrain = dataname + 'trainset%d.txt' % num_train
    with open(dir + filetest, 'wb') as ftest:
        pickle.dump(dict_test, ftest)
    with open(dir + filetrain, 'wb') as ftrain:
        pickle.dump(dict_train, ftrain)
    print("%d samples in Testset; %d samples in Trainset." % (num_test, num_train))
    return


if __name__ == '__main__':
    # Load data from directory
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type=str, default="hp19")
    parser.add_argument("--testsetSize", type=int, default=2000)
    args = parser.parse_args()
    datapath = args.dir
    dataname = datapath.upper()
    test_set_size = args.testsetSize
    dir = "dataset/"
    filenames = []
    for (_, _, f) in walk(datapath):
        filenames.extend(f)
        # eg. filename = 'HHHHHHHHHHHHPHPPPHP.conf'
        # eg. raw_conf = '[(0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (0, 3), (-1, 3), (-2, 3),...,...]'

    # data augmentation of one HP sequence and its reverse sequence
    raw_confs = []
    raw_confs_reverse = []
    input_HPs = []
    input_HPs_reverse = []
    dict0 = {'H': 0, 'P': 1}
    print('number of filenames：', len(filenames))
    _ = 0
    for filename in filenames:
        if filename[1] != '_':
            input_HP = []
            for hp in filename:
                if hp == '.':
                    break
                input_HP.append(dict0[hp])
            input_HPs.append(input_HP)
            input_HPs_reverse.append(input_HP[::-1])
            with open(datapath + '/' + filename, "r") as raw_conf:
                for line in raw_conf:  # 实际loop只有一次
                    conf, conf_reverse = formalize_data(line)
                    raw_confs.append(conf)
                    raw_confs_reverse.append(conf_reverse)
        _ += 1
        if _ % 1000 == 0:
            print(len(input_HPs))
    input_HPs.extend(input_HPs_reverse)
    raw_confs.extend(raw_confs_reverse)
    output_confs = raw_confs
    # output_confs: values [-1,0,1], dtype= int , len= 13454*2 = 26908
    # input_HPs: valuse [0, 1], dtype= np.float64, len= 26908

    # 2. wash data
    # To ensure no repeats unless sequence and its reversed one.
    washed_inputHPs = []
    washed_outputconfs = []
    for i in range(len(input_HPs)):
        if i == 0:
            washed_inputHPs.append(input_HPs[i])
            washed_outputconfs.append(output_confs[i])
        else:
            for j in range(len(washed_inputHPs)):
                if input_HPs[i] == washed_inputHPs[j]:
                    print(i, 'Seq Repeated')
                    break
                if j == len(washed_inputHPs) - 1:
                    washed_inputHPs.append(input_HPs[i])
                    washed_outputconfs.append(output_confs[i])
    print(len(washed_inputHPs))

    # 3. dump whole datasets into directory
    keys = ['num_of_sample', 'chain_length', 'input_HPs', 'output_confs']
    values = [len(washed_inputHPs), 19, washed_inputHPs, washed_outputconfs]
    d = zip(keys, values)
    data_dict = dict(d)
    with open(dir + dataname + 'mer_conf_data.txt', 'wb') as f:
        pickle.dump(data_dict, f)

    #
    wash_data(dir)
    creat_training_data(test_set_size)




