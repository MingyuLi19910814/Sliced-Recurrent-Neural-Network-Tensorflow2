MAX_LEN = 512
EMBEDDING_DIM = 200
# this value is fixed since we are using pre-trained GloVe word embeddings
GloVe_PATH = './data/glove.6B.200d.txt'

DATA_PATH = {
    'Yelp2013': './data/yelp_2013.csv',
    'Yelp2014': './data/yelp_2014.csv',
    'Yelp2015': './data/yelp_2015.csv'
}

TFRECORD_DIR = './tfrecord'
import os
MODEL_DIR = './saved_model'
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.ckpt')