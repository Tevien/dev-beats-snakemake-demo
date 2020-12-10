import tensorflow as tf
import argparse
import pickle as pkl
from utils import preprocess_data

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int,
                    default=5000)
parser.add_argument('--output', type=str,
                    default="mnist.pkl")

args = parser.parse_args()

# Get input
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Cut dataset to required size
if args.size:
    x_train = x_train[:args.size]
    y_train = y_train[:args.size]
    x_test = x_test[:args.size]
    y_test = y_test[:args.size]

# Save output
output = dict()
output['x_train'] = preprocess_data(x_train)
output['x_test'] = preprocess_data(x_test)
output['y_train'] = y_train
output['y_test'] = y_test

with open(args.output, 'wb') as f:
    pkl.dump(output, f)
