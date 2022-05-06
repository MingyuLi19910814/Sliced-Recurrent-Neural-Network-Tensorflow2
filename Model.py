import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.layers import Embedding, Dense, GRU, Input
from SimplifiedTimeDistributed import ModelTimeDistributed
from constants import EMBEDDING_DIM, MAX_LEN, GloVe_PATH

def create_model(args, vocab=None):
    print("Loading GloVe embeddings")
    if not os.path.isfile(GloVe_PATH):
        raise FileNotFoundError("Can not find glove.6B.200d.txt in ./data. Please download from "
                                "https://nlp.stanford.edu/projects/glove/ and save to ./data")

    if vocab is not None:
        embeddings_index = {}
        with open(GloVe_PATH) as f:
            for line in tqdm(f.readlines()):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print('Finish Loading GloVe embeddings')

        # use pre-trained GloVe word embeddings to initialize the embedding layer
        embedding_matrix = np.random.random((args.max_num_words + 1, EMBEDDING_DIM))
        for word, i in vocab.items():
            if i < args.max_num_words:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be random initialized.
                    embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(args.max_num_words + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_LEN / 64,
                                    trainable=True)
    else:
        embedding_layer = Embedding(args.max_num_words + 1,
                                    EMBEDDING_DIM,
                                    input_length=MAX_LEN / 64,
                                    trainable=True)
    if args.model == 'srnn':
        # (-1, 8)
        input1 = tf.keras.layers.Input(shape=(MAX_LEN // 64), dtype=tf.int32)
        # (-1, 8, EMBEDDING_DIM)
        embed1 = embedding_layer(input1)
        # (-1, num_filters)
        gru1 = tf.keras.layers.GRU(args.num_filters,
                                   return_sequences=False,
                                   activation=None,
                                   recurrent_activation='sigmoid')(embed1)
        encoder1 = tf.keras.Model(input1, gru1)
        # (-1, 8, 8)
        input2 = tf.keras.layers.Input(shape=(8, MAX_LEN // 64), dtype=tf.int32)
        # (-1, 8, num_filters)
        embed2 = ModelTimeDistributed(encoder1)(input2)
        # (-1, num_filters)
        gru2 = tf.keras.layers.GRU(args.num_filters,
                                   return_sequences=False,
                                   activation=None,
                                   recurrent_activation='sigmoid')(embed2)
        encoder2 = tf.keras.Model(input2, gru2)
        # (-1, 8, 8, 8)
        input3 = tf.keras.layers.Input(shape=(8, 8, MAX_LEN // 64), dtype=tf.int32)
        # (-1, 8, num_filters)
        embed3 = ModelTimeDistributed(encoder2)(input3)
        # (-1, num_filters)
        gru3 = tf.keras.layers.GRU(args.num_filters,
                                   return_sequences=False,
                                   activation=None,
                                   recurrent_activation='sigmoid')(embed3)
        # (-1, num_class)
        pred = tf.keras.layers.Dense(args.num_class, activation='softmax', )(gru3)
        model = tf.keras.Model(input3, pred)
    else:
        # (-1, 8, 8, 8)
        inputs = tf.keras.layers.Input(shape=(8, 8, MAX_LEN // 64), dtype=tf.int32)
        # (-1, MAX_LEN)
        input_flatten = tf.reshape(inputs, (-1, MAX_LEN))
        # (-1, MAX_LEN, EMBEDDING_DIM)
        embed = embedding_layer(input_flatten)
        # (-1, EMBEDDING_DIM)
        gru = tf.keras.layers.GRU(args.num_filters,
                                  return_sequences=False,
                                  recurrent_activation='sigmoid',
                                  activation=None)(embed)
        # (-1, num_class)
        pred = tf.keras.layers.Dense(args.num_class, activation='softmax')(gru)
        model = tf.keras.Model(inputs, pred)
        return model
    return model