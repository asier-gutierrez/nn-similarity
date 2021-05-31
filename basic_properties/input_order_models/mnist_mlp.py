import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


def train():
    model = tf.keras.Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    # np.random.seed(analysis_value)
    train_idxs = np.random.permutation(len(x_train))
    test_idxs = np.random.permutation(len(x_test))

    # Train randomization
    x_train = x_train[train_idxs]
    y_train = y_train[train_idxs]

    # Test randomization
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]

    # Categories
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=20,
                        verbose=1,
                        validation_split=0.1)
    return model
