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

    # build model
    # (None, 8)
    input1 = Input(shape=(MAX_LEN // 64,), dtype=tf.int32)
    # (None, 8, EMBEDDING_DIM)
    embed = embedding_layer(input1)
    # (None, NUM_FILTERS), sequence_dim is removed since return_sequences=False
    gru1 = GRU(args.num_filters, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed)
    # input: (None, 8)
    # output: (None, NUM_FILTERS)
    Encoder1 = tf.keras.Model(input1, gru1)

    # (None, 8, 8)
    input2 = Input(shape=(8, MAX_LEN // 64,), dtype=tf.int32)
    # (None, 8, NUM_FILTERS)
    embed2 = ModelTimeDistributed(Encoder1)(input2)
    # (None, NUM_FILTERS)
    gru2 = GRU(args.num_filters, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed2)
    # input: (None, 8, 8)
    # output: (None, NUM_FILTERS)
    Encoder2 = tf.keras.Model(input2, gru2)

    # (None, 8, 8, 8)
    input3 = Input(shape=(8, 8, MAX_LEN // 64), dtype=tf.int32)
    # (None, 8, NUM_FILTERS)
    embed3 = ModelTimeDistributed(Encoder2)(input3)
    # (None, NUM_FILTERS)
    gru3 = GRU(args.num_filters, recurrent_activation='sigmoid', activation=None, return_sequences=False)(embed3)
    # (None, 5)
    preds = Dense(args.num_class, activation='softmax')(gru3)
    model = tf.keras.Model(input3, preds)
    return model