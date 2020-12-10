import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import argparse
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    default="mnist.pkl")
parser.add_argument('--output', type=str,
                    default="model.h5")
args = parser.parse_args()

# Load the data from the previous step
with open(args.input, 'rb') as f:
    in_dict = pkl.load(f)

shape = in_dict['x_train'].shape[1:]
input_shape = (shape[0], shape[1], 1)

# Make model
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

# Train model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=in_dict['x_train'], y=in_dict['y_train'], epochs=10)

# Save model
model.save(args.output)
