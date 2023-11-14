import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

import pickle
import numpy as np

print(tf.__version__)


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Training data
directory_base_address = r'C:\Users\HP\PycharmProjects\ex3\cifar-10-python\cifar-10-batches-py/'
X = []
Y = []
for i in range(1, 6):
    datadict = unpickle(directory_base_address + '/data_batch_' + str(i))
    X.append(datadict["data"])
    Y.append(datadict["labels"])

X_train = np.concatenate(X)
Y_train = np.concatenate(Y)

X_train = X_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint32")

# Test data
datadict = unpickle(directory_base_address + '/test_batch')
X_test = datadict["data"]
Y_test = datadict["labels"]
labeldict = unpickle(directory_base_address + '/batches.meta')
label_names = labeldict["label_names"]
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint32")
Y_test = np.array(Y_test)

X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = tf.one_hot(Y_train.astype(np.int32), depth=10)
Y_test = tf.one_hot(Y_test.astype(np.int32), depth=10)

model = Sequential()

# model.add(tf.keras.layers.Flatten())
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(128, activation=tf.nn.sigmoid))
model.add(Dense(64, activation=tf.nn.sigmoid))
model.add(Dense(32, activation=tf.nn.sigmoid))
model.add(Dense(16, activation=tf.nn.sigmoid))
model.add(Dense(10, activation=tf.nn.sigmoid))

print(model.summary())

opt = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=100, verbose=1, batch_size=64,
                    validation_split=0.3)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Accuracy:", test_acc)

# summary of history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'accuracy'], loc='lower right')
plt.show()

# Comapring with others result - 1NN and bayesian are from previous exercise
accuracy_list = [test_acc * 100, 9.61, 22.79, 28.85, 36.14, 38.32, 39.62, 43.35, 36.23]
different_classifier = ['Neural Net', '1-NN', 'NB[1x1]', 'NB[2x2]', 'NB[4x4]', 'NB[6x6]', 'NB[8x8]', 'NB[16x16]', 'NB[32x32']
accuuracy_bar = accuracy_list

plt.bar(different_classifier, accuuracy_bar)
plt.title('Accuracy of various classifier')
plt.xlabel('Different classifier')
plt.ylabel('Accuracy %')
plt.show()
