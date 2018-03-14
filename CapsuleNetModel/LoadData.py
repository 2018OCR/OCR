import numpy as np
import tensorflow as tf
from keras.datasets import mnist


def load_mnist(batch_size, is_training=True):
    # path = os.path.join('data', 'mnist')
    (trainX, trainY), (teX, teY) = mnist.load_data()
    if is_training:
        trainX = trainX.reshape((60000, 784)).astype(np.float32)

        trainY = trainY.reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        teX = teX.reshape((10000, 784)).astype(np.float)

        teY = teY.reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return X, Y