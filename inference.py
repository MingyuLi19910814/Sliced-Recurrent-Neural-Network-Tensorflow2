import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import argparse
from dataset import TestDataset
from Model import create_model
from constants import BEST_MODEL_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path of the file containing the test sentences')
parser.add_argument('--output_path', type=str, help='path to the file to save the output prediction')
parser.add_argument('--model_path', type=str, help='')
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--max_num_words', type=int, default=30000)
parser.add_argument('--num_class', type=int, default=5)
parser.add_argument('--num_filters', type=int, default=50, help='hidden size of the RNN')

def inference(args):
    text_sequence = TestDataset(args).get_text_sequence()
    model = create_model(args, None)
    model.load_weights(BEST_MODEL_PATH)
    pred = model.predict(text_sequence)
    pred = np.argmax(pred, axis=-1) + 1
    with open(args.output_path, 'w') as f:
        for p in pred:
            f.write('{}\n'.format(p))


if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)