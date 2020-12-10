import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import load_model

import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    default="model.h5")
parser.add_argument('--index', type=int,
                    default=9)
parser.add_argument('--data', type=str,
                    default="mnist.pkl")
args = parser.parse_args()

model = load_model(args.input)

image_index = args.index
# Load the data from the previous step
with open(args.data, 'rb') as f:
    in_dict = pkl.load(f)
data = in_dict['x_test']
label = in_dict['y_test'][image_index]

pred = model.predict(data[image_index].reshape(1, data.shape[1], data.shape[2], 1)).argmax()

padding = 5
fig, ax = plt.subplots(1)
plt.ioff()
ax.imshow(data[image_index].reshape(data.shape[1], data.shape[2]), cmap='Greys')
ax.axis("off")
ax.annotate(
    s='Prediction: %s \nTruth: %s' % (pred, label),
    fontsize=12,
    xy=(0, 0),
    xytext=(padding-1, -(padding-1)),
    textcoords='offset pixels',
    bbox=dict(facecolor='white', alpha=1, pad=padding),
    va='top',
    ha='left',
    )
plt.savefig("test.png")
