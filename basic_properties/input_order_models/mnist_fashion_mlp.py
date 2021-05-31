import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


def train(model_tl):
    x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(
        model_tl.layers[9].output)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='softmax')(x)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
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
    model = tf.keras.Model(inputs=model_tl.inputs, outputs=x)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=256,
                        epochs=20,
                        verbose=1,
                        validation_split=0.1)

    return model


def prepare():
    model_tl = tf.keras.Sequential()
    model_tl.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                        input_shape=(28, 28, 1)))
    model_tl.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model_tl.add(MaxPooling2D((2, 2)))
    model_tl.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_tl.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_tl.add(MaxPooling2D((2, 2)))
    model_tl.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_tl.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model_tl.add(MaxPooling2D((2, 2)))
    model_tl.add(Flatten())
    model_tl.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model_tl.add(Dropout(0.2))
    model_tl.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
    model_tl.add(Dropout(0.2))
    model_tl.add(Dense(10, activation='softmax'))
    model_tl.summary()
    model_tl.compile(loss='categorical_crossentropy',
                     optimizer=RMSprop(),
                     metrics=['accuracy'])
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    history = model_tl.fit(x_train, y_train,
                           batch_size=256,
                           epochs=20,
                           verbose=1,
                           validation_split=0.1)

    for layer in model_tl.layers[:10]:
        layer.trainable = False

    return model_tl