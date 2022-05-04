import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from dataset import TrainDataset
from Model import create_model
from constants import BEST_MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--create_tfrecord', action='store_true',
                    help='''
                    Should be only called once for every dataset.
                    If set, the program would load data from csv specified by "--dataset", "--train_data_path" and
                    "--test_data_path", create tfrecord and save to './tfrecord/train.tfrecord",
                    "./tfrecord/val.tfrecord," and "./tfrecord/test.tfrecord".
                    Otherwise, the program would ignore "--dataset", "--train_data_path" and "--test_data_path" and
                    load directly from "./tfrecord/train.tfrecord", "./tfrecord/val.tfrecord", "./tfrecord/test.tfrecord".
                    ''')
parser.add_argument('--dataset', type=str, default='Yelp2013',
                    choices=['Yelp2013', 'Yelp2014', 'Yelp2015', 'Amazon_F', 'Custom'], help='dataset to train on')
parser.add_argument('--train_data_path', type=str, default='',
                    help='if dataset is "Custom", please specify the path to the train csv')
parser.add_argument('--test_data_path', type=str, default='',
                    help='if dataset is "Custom", please specify the path to the test csv')
parser.add_argument('--val_size', type=float, default=0.1, help='validation size')
parser.add_argument('--test_size', type=float, default=0.1, help='test size if test_csv is not specified')
parser.add_argument('--rnn_type', type=str, default='GRU', choices=['SimpleRNN', 'LSTM', 'RNN'],
                    help='rnn cell to use (default GRU)')
parser.add_argument('--num_filters', type=int, default=50, help='hidden size of the RNN')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--max_num_words', type=int, default=30000)
parser.add_argument('--num_class', type=int, default=5)
parser.add_argument('--epochs', type=int, default=10)


def train(args):
    dataset = TrainDataset(args)
    train_ds = dataset.get_datasets('train')
    val_ds = dataset.get_datasets('val')
    test_ds = dataset.get_datasets('test')
    vocabulary = dataset.get_vocabulary()
    del dataset
    model_callback = tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, save_weights_only=True,
                                                        verbose=1, monitor='val_acc', mode='max')
    model = create_model(args, vocabulary)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['acc'])
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=[model_callback])
    del model, train_ds, val_ds

    best_model = create_model(args, None)
    best_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=['acc'])
    best_model.load_weights(BEST_MODEL_PATH)
    print(best_model.evaluate(test_ds))


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
