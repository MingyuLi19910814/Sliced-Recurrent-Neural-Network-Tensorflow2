# Overview

This repository implements the Sliced Recurrent Neural Networks of [Zeping Yu](https://arxiv.org/pdf/1807.02291v1.pdf)
in Tensorflow 2. It's based on the original [Keras implementation](https://github.com/zepingyu0512/srnn). 

# My Contribution
1. Implementation in Tensorflow 2 and Python 3
2. Implements TimeDistributed Module which has problem working with GRU in current Tensorflow version.
3. Efficient data processing pipeline
4. Training pipeline of Yelp2013, Yelp2014, Yelp25 and custom dataset
5. Inference pipeline

# Requirement
It's strongly recommended to use Anaconda to build the environment.
```commandline
conda env create -f environment.yml
conda activate tensorflow
```

# Train

Everytime training on a new dataset, remember to add the **--create_tfrecord** option
to create tfrecords from the dataset. Otherwise, it would use old tfrecords saved in **./tfrecord**
which means still training on old dataset. Once the tfrecords are created, you could 
skip it next time.

The model with highest evaluation accuracy would be saved in **./saved_model**
## Download Glove Vector
This implementation uses GloVe embedding as the pretrained embedding weights. 
Please download **glove.6B.200d.txt** from [this link](https://www.kaggle.com/datasets/incorpes/glove6b200d/code)
and save as **./data/glove.6B.200d.txt**.

## Train on Yelp2013
Download Yelp2013 dataset from [this link](https://figshare.com/articles/dataset/Yelp_2013/6292142) and save as **./data/yelp_2013.csv**
```commandline
python train.py --dataset Yelp2013 \
                --val_size 0.1 \
                --test_size 0.1 \
                --epochs 10 \
                --create_tfrecord \
                --batch_size 2048 #RTX2080ti

```

## Train on Yelp2014
Download Yelp2014 dataset from [this link](https://figshare.com/articles/dataset/Untitled_Item/6292253) and save as **./data/yelp_2014.csv**
```commandline
python train.py --dataset Yelp2014 \
                --val_size 0.1 \
                --test_size 0.1 \
                --epochs 10 \
                --create_tfrecord \
                --batch_size 2048
```

## Train on Yelp2015
Download Yelp2015 dataset from [this link](https://figshare.com/articles/dataset/Yelp_2015/6292334) and save as **./data/yelp_2015.csv**
```commandline
python train.py --dataset Yelp2015 \
                --val_size 0.1 \
                --test_size 0.1 \
                --epochs 10 \
                --create_tfrecord \
                --batch_size 2048
```

## Train on custom dataset
Please refer to **data/train_sample.csv** to prepare the custom data.  
The train dataset should be a csv file containing two columns.
1. The first column contains the label ranging from 1 to class_num
2. The second column contains the sentences.
```commandline
python train.py --dataset Custom \ # This must be "Custom"
                --train_data_path ./data/train_sample.csv \
                --val_size 0.1 \
                --test_size 0.1 \
                --epochs 10\
                --create_tfrecord \
                --batch_size 2048
```

# Result

Yelp 2013, batch size = 2048: 0.650
Yelp 2014, batch size = 2048: 0.689
Yelp 2015, batch size = 2048: 0.7195
