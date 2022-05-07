# HPprotein-prediction-SCN
HPprotein-prediction-SCN is a repo providing several python scripts of deep neural network models for the 2D HP lattice protein prediction problem, based on Tensorflow 1.x. 

In this repository, we provide examples of  **S**trongly-**C**orrelated **N**etwork with **A**ttention **M**echanism (**SCN-AM**) model and the **B**ig-**S**tride **R**epresentation (BSR) method to solve the protein folding prediction problems. The molecular chains here are 19mer theoretical proteins of two-dimensional lattice HP model. For comparison, we also provide several standard NN models on the same task to show the improvement brought by SCN-AM and BSR. 

## Model Illustration
In this section, we briefly introduce the mechanism of our own model that has been employed in `AttentionNN_HPSCC.py` and our own version of Conditional Random Field(CRF) method.
> More Details are Shown in Paper(https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

We provide a data processing script to produce and partition training sets and test sets of HP 19mer chains. The repository [vvoelz/HPSandbox](https://github.com/vvoelz/HPSandbox) provides the raw data of HP proteins.

## Repository Contents
+ Network Model Scripts:
  1. [`baseline_CNNmodel.py`](baseline_CNNmodel.py)：A model with a standard two-layer Convolutional Neural Network.
  2. [`CRFmodel.py`](CRFmodel.py)：A model utilizing Conditional Random Field method for protein structure prediction, in our own way. For more details about CRF, see [Sheng Wang, et.al](https://www.nature.com/articles/srep18962). This model uses the same CNN module as `baseline_CNNmodel.py`.
  3. [`SCN_CNNmodel.py`](SCN_CNNmodel.py)：A model employs Strongly-Correlated Network and the model achitecture is partly based on `baseline_CNNmodel.py`. It builds a strong correlation among protein residues in physical space through a self-consistent iteration loop.
  4. [`AttentionNN_HPSCC.py`](AttentionNN_HPSCC.py)：A model combining SCN mechanism and our own designed network block with self-attention layers. This model exhibits the optimum predicting performance. 
  ps. For all the model scripts, use `python *.py --help` command to check all the optional parameters.

+ [`ConfMapper.py`](ConfMapper.py): ConfMapper script realized one of the key algorithm, **BSR**, in this package. This algorithm group several small residues together form a big stride representation of the output structure. All the model scripts except for `CRFmodel.py` in this package can use `ConfMapper.py` directly.(CRF is not compatible with BSR) BSR is set not enabled by default. And thus for each model script, the optional parameter `-bn, --base_num` is set as 1. To enable the usage of BSR, run the script with `python *Model*.py -bn n`. Here `n` should be a positive integer which sets the stride size of BSR to `n`.

+ [`/dataset`](/dataset) & [`dataprocessor.py`](dataprocessor.py): The directory is used to store the training set and test set of HP 19mer proteins' sequence and folding structure. Two dataset files in the directory are processed and randomly partitioned. For customized usage of original data [HPSandbox](https://github.com/vvoelz/HPSandbox)， we privide a `dataprocessor` script. (eg. validation set, cross validation, process of other HP protein datas etc.) Please refer to [Dataset Generation](#Dataset-Generation) section for more details.

+ [`/data`](/data): The accuracy data file of each model script is defaultly stored in this directory.

## Model Usage
The four neural network model python scripts contain: model itself, basic components of network models and accuracy metric. These modules have been encapsulated and can be imported from the scripts under the working directory. 

You can also run these model scripts directly and get a complete model training process with period of 1200 epoches defaultly. We take `baseline_CNNmodel` as an example. We can use this command to run a model script in CMD or Shell:  
```sh
$ python baseline_CNNmodel.py
```
  This operation builds a corresponding neural network and optimize its variables on the training set. And during the running time, we can get the following statistics of in time model performance.
  > Step 2900, Train Accuracy Distribution:\
  > [0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.005, 0.008, 0.012, 0.017, 0.021, 0.047, 0.048, 0.093, 0.092, 0.119, 0.153, 0.143, 0.129, 0.111]\
  > Train Cross Entropy: 9.664868\
  > Step 2900, Test Accuracy Distribution:\
  > [0.0, 0.0, 0.0, 0.0005, 0.002, 0.002, 0.006, 0.009, 0.021, 0.028, 0.039, 0.066, 0.078, 0.105, 0.1265, 0.134, 0.1215, 0.1195, 0.0825, 0.0595]\
  > \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \*\
  > Test set latest accuracies:\
  > [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005, 0.0005, 0.001, 0.003, 0.0035, 0.002, 0.01, 0.0125, 0.0105, 0.02, 0.023, 0.021, 0.0215, 0.0315, 0.036, 0.0405, 0.045,   0.0525, 0.0535, 0.0575, 0.0595]

As shown above, for every 100 steps, the model prints the accuracy and the cross entropy (loss) on 1000 randomly chosen training samples and accuracy on the whole test set to *stdout*. Accuracy here is shown as a discrete distribution, which has 19 values. Each value denotes the the propotion of results in which model predicts correctly on exactly n coordinates of one chain folding. And thus, the last value of the accuracy distribution means the proportion that the model has  correctly predicted all coordinates. 
    
We define this last value as the prediction accuracy of the model，and record it every 100 steps. These data are written into a *txt* file in binary form using python package `pickle`, named as `baselineCNN_epoch*basenum*_name**.txt` and stored in `data\data_baselineCNN`. To monitor the change of the model performance during the optimization, *Test set latest accuracies* shows accuracy of the model in last 5000 steps. 


### Commandline Parameters
To customize the hyperparameters of model structures, training process and usage of ConfMapper etc., we can take optional input arguments while running model scripts.
```
python SCN_CNNmodel.py -nn 1 -use_GPU 0,1 -bn 5
```
To see all the optional arguments and their default values, please use the following command:
```
python SCN_CNNmodel.py --help
```
> usage: SCN_CNNmodel.py [-h] [-b BATCH_SIZE] [-nn NAME_NUM] [-bn BASE_NUM] \
>                        [-in ITER_NUM] [-en EPOCH_NUM] [-use_GPU USE_GPU]
>
> default configuration: batch_size = 60 c_length = 19 num_train = 11470 \
> embedding_size = 128 epoches = int(num_train / batch_size) \
> num_training_steps = epoches * 500
>
> optional arguments: \
> -h, --help            show this help message and exit \
> -b BATCH_SIZE, --batch_size BATCH_SIZE \
> -nn NAME_NUM, --name_num NAME_NUM \
> -bn BASE_NUM, --base_num BASE_NUM \
> -in ITER_NUM, --iter_num ITER_NUM \
> -en EPOCH_NUM, --epoch_num EPOCH_NUM \
> -use_GPU USE_GPU 

### Usage for tensorflow 2.x
This repository is written using tensorflow 1.15. If you want to use it under tensorflow 2.0 framework, please replace the `import tensorflow as tf` command at the beginning  of each model script with:
```
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
```
.
## Dataset Generation
In this section, we'll introduce how to use [`dataprocessor.py`](dataprocessor.py) to process the original data in the [HPSandbox](https://github.com/vvoelz/HPSandbox) and generate train/test/validation sets. 

We take HP 19mer protein as an example: \
Download and decompress [hp19.tar.gz](https://github.com/vvoelz/HPSandbox/blob/master/sequences/conf/hp19.tar.gz) under the working directory. The file in `/hp19` has the name denoting the HP protein sequence and content in the file notating two-dimensional coordinates of the folding structure:

> filename = 'HHHHHHHHHHHHPHPPPHP.conf' \
> Configuration = [(0, 0), (0, 1), (1, 1), (1, 2), (0, 2), (0, 3), (-1, 3), (-2, 3),...,...]
     
Then, simply use this command in bash or CMD:
```
python dataprocessor.py -dir hp19 -testsetSize 2000
```
This command will read files from directory `hp19` and generate train/test file: [HP19testset2000.txt](dataset/HP19testset2000.txt) & [HP19trainset11470.txt](dataset/HP19trainset11470.txt) in `\dataset` directory. `-dir` parameters denotes raw HP protein data is in which directory and `-testset_size 2000` will partition a test set with 2000 samples. Other files in `\dataset` directory are  byproducts of this script. You can read the note in the script for more details. This script contains procedures to check whether there are repeated sequences in the dataset. So the whole process may take several minutes. 


