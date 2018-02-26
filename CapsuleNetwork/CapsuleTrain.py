from keras import utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from ChnRec.CapsuleNetModel import model
import numpy as np
import pylab

# Arguments
batch_size = 100
num_classes = 2500
img_rows, img_cols = 65, 64

x_train = np.loadtxt('chn_num.txt')
y_train = np.loadtxt('chn_num_label.txt')
x_test = np.loadtxt('chn_num_test.txt')
y_test = np.loadtxt('chn_num_label_test.txt')

x_train = x_train.reshape(-1, img_rows, img_cols, 1)
x_test = x_test.reshape(-1, img_rows, img_cols, 1)
x_train /= 255.0
x_test /= 255.0

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = utils.to_categorical(y_train, 28)
y_test = utils.to_categorical(y_test, 28)


class MyCallback(Callback):
    def __init__(self):
        self.epoch_cnt = 0

    def on_epoch_end(self, epoch, logs=None):
        model_dir = 'model_file/chn_num_model.h5'
        model.save_weights(model_dir)
        self.epoch_cnt += 1


callbacker = MyCallback()

if __name__ == '__main__':
    # print(x_test.shape)
    # print(y_test[0])
    # print(x_train.shape)
    # print(y_train.shape)
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=3,
              verbose=1,
              callbacks=[callbacker],
              validation_data=(x_test, y_test))

