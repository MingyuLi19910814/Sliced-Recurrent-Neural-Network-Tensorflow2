import numpy as np
import tensorflow as tf
import os
import pandas as pd
from tqdm import tqdm
import shutil
import pickle
from constants import MAX_LEN,DATA_PATH, TFRECORD_DIR


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class TrainDataset:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.train_record_path = os.path.join(TFRECORD_DIR, 'train.tfrecord')
        self.val_record_path = os.path.join(TFRECORD_DIR, 'val.tfrecord')
        self.test_record_path = os.path.join(TFRECORD_DIR, 'test.tfrecord')
        self.tokenizer_path = os.path.join(TFRECORD_DIR, 'tokenizer.bin')
        if args.create_tfrecord:
            print('reading data')
            if args.dataset == 'Custom':
                assert os.path.isfile(args.train_data_path) and os.path.isfile(args.test_data_path)
                train_df, val_df = self.split_csv(args.train_data_path, [1 - args.val_size, args.val_size])
                test_df = pd.read_csv(args.test_data_path)
                train_data = train_df.to_numpy()
                val_data = val_df.to_numpy()
                test_data = test_df.to_numpy()
            else:
                '''
                for Yelp dataset, only the train csv is provided. To make the process same as custom dataset,
                we load the csv, split into train-val-test and save to './data/tmp'
                '''
                assert os.path.isfile(DATA_PATH[args.dataset])
                val_df, test_df, train_df = \
                    self.split_csv(DATA_PATH[args.dataset], [args.val_size, args.test_size, 1 - args.val_size - args.test_size])
                train_data = self.preprocess_yelp(train_df)
                val_data = self.preprocess_yelp(val_df)
                test_data = self.preprocess_yelp(test_df)
            del train_df, val_df, test_df
            print('     Done')
            print('start fitting tokenizer...')
            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=args.max_num_words)
            tokenizer.fit_on_texts(texts=train_data[:, 1])
            print('     Done')
            shutil.rmtree(TFRECORD_DIR, ignore_errors=True)
            os.makedirs(TFRECORD_DIR, exist_ok=False)
            self.create_tfrecord(train_data, self.train_record_path, tokenizer)
            self.create_tfrecord(val_data, self.val_record_path, tokenizer)
            self.create_tfrecord(test_data, self.test_record_path, tokenizer)
            with open(self.tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            del train_data, val_data, test_data, tokenizer
        self.record_path = {
            'train': self.train_record_path,
            'val': self.val_record_path,
            'test': self.test_record_path
        }
        assert os.path.isfile(self.train_record_path) \
               and os.path.isfile(self.val_record_path) \
               and os.path.isfile(self.test_record_path) \
               and os.path.isfile(self.tokenizer_path)


    def get_datasets(self, split='train'):
        assert split in self.record_path

        def decode_fn(example):
            feature = tf.io.parse_single_example(
                example,
                features={
                    'sequence': tf.io.FixedLenFeature([], dtype=tf.string),
                    'label': tf.io.FixedLenFeature([], dtype=tf.int64),
                }
            )
            sequence = tf.io.decode_raw(feature['sequence'], tf.int32)
            # sequence = tf.reshape(sequence, (-1, 512))
            #
            # padded_seqs_split = np.zeros(shape=(8, 8, 8), dtype=np.float32)
            # # (8, ndarray(64))
            # split1 = np.split(sequence, 8)
            # for j in range(8):
            #     # (8, ndarray(8))
            #     s = np.split(split1[j], 8)
            #     padded_seqs_split[j] = np.split(split1[j], 8)

            return tf.reshape(sequence, (8, 8, 8)), feature['label']

        dataset = tf.data.TFRecordDataset([self.record_path[split]]).map(decode_fn).batch(self.batch_size)
        return dataset

    def get_vocabulary(self):
        with open(self.tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer.word_index


    def create_tfrecord(self, data, save_path, tokenizer):
        with tf.io.TFRecordWriter(save_path) as writer:
            label = data[:, 0].astype(np.int32)
            text = data[:, 1]
            text_ids = tokenizer.texts_to_sequences(text)
            text_pad = tf.keras.preprocessing.sequence.pad_sequences(text_ids, maxlen=MAX_LEN).astype(np.int32)
            for idx in range(label.shape[0]):
                feature = {
                    'sequence': _bytes_features(text_pad[idx].tobytes()),
                    'label': _int64_features(label[idx]),
                }
                msg = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
                writer.write(msg)

    def split_csv(self, csv_path, sizes):
        assert (isinstance(sizes, list) or isinstance(sizes, tuple)) and np.isclose(np.sum(sizes), 1)
        csv_data = pd.read_csv(csv_path, index_col=0)
        csv_data = csv_data.sample(frac=1, random_state=0)
        sizes = np.cumsum(sizes)
        i0 = 0
        dfs = []
        for s in sizes:
            i1 = int(s * csv_data.shape[0])
            dfs.append(csv_data.iloc[i0:i1])
            i0 = i1
        return dfs

    def preprocess_yelp(self, df):
        # the class index of yelp dataset start from 1, we need to make it start from 0
        label = df['stars'].to_numpy() - 1
        text = df['text'].to_numpy()
        data = np.concatenate([label[:, np.newaxis], text[:, np.newaxis]], axis=1)
        return data


