# HPprotein-prediction-SCN
HPprotein-prediction-SCN is a package provides several python scripts of Deep Neural Network Models for 2D HP lattice protein fold prediction, based on Tensorflow 1.x. 

In this repository, we provide examples of **S**trong **C**orrelated **N**etwork(**SCN**) model and **Configuration Mapping** method to solve the protein folding prediction problems. The molecular chains here are 19mer theoretical proteins of two-dimensional lattice HP model. For comparision, we also provide several standard NN models on the same task to show the improvement brought by SCN and CM. 
> More Details are Shown in Paper(links)

In each python script, we realize the construction and training process of the neural network models. and also the evaluation of predicting accuracies, based on Tensorflow 1.15.
We provide a data making script to produce and partition training sets and data sets of HP 19mer chains. The repository [vvoelz/HPSandbox](https://github.com/vvoelz/HPSandbox) provides the raw data of HP proteins.

## LICENSE
pass

## Repository Contents
+ Four neural network models python scripts containing: model itself, basic components of network models and accuracy metric. These functions can be imported from the scripts under working directory. You can also run these model scripts directly and get an example of defaultly 1200 epoches training process. \
  eg. `python baseline_CNNmodel.py -[optional parameters]` 
  > Step 2900, Train Accuracy Distribution:\
  > [0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.005, 0.008, 0.012, 0.017, 0.021, 0.047, 0.048, 0.093, 0.092, 0.119, 0.153, 0.143, 0.129, 0.111]\
  > Train Cross Entropy: 9.664868\
  > Step 2900, Test Accuracy Distribution:\
  > [0.0, 0.0, 0.0, 0.0005, 0.002, 0.002, 0.006, 0.009, 0.021, 0.028, 0.039, 0.066, 0.078, 0.105, 0.1265, 0.134, 0.1215, 0.1195, 0.0825, 0.0595]\
  > \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \* \*\
  > Test set latest accuracies:\
  > [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005, 0.0005, 0.001, 0.003, 0.0035, 0.002, 0.01, 0.0125, 0.0105, 0.02, 0.023, 0.021, 0.0215, 0.0315, 0.036, 0.0405, 0.045,   0.0525, 0.0535, 0.0575, 0.0595]

    As shown in above, `model.py` trains a neural network and monitors its 
    (The illustration holds the same for all the model script in the following list.)
1. .py
2. .py
3. .py

## How to use this repository

### Installation

### Model Usage

### Dataset Generation

## Model Illustration
(可能这一段应该放到最开始的地方)
In this section, we briefly introduce the mechanism of our own model AttentionNN-HPSCC and our own version of Conditional Random Field.


[Test Something interesting](#how-to-use-this-repo)
> This is the first level of quotation:
> > I'd like to show that there is no evidence that we can't quote 2nd time in a quotation field.
> > > This is a rather interesting mechanism about quotation iteration.

    This is a code block paragraph(To build a code block, I just need to type 4 spaces or one Tab at the beginning)
    import fxxxain as fx

