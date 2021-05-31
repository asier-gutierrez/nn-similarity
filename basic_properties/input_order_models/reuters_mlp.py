import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


def train():
    model = tf.keras.Sequential()
    model.add(
        Dense(512, activation='relu', input_shape=(10000,), kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(46, activation='softmax'))
    # the data, split between train and test sets
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    # Vectorized training data
    x_train = vectorize_sequences(train_data)
    # Vectorized test data
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    train_idxs = np.random.permutation(len(x_train))
    test_idxs = np.random.permutation(len(x_test))

    # Train randomization
    x_train = x_train[train_idxs]
    y_train = y_train[train_idxs]

    # Test randomization
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]

    # Categories
    y_train = tf.keras.utils.to_categorical(y_train, 46)
    y_test = tf.keras.utils.to_categorical(y_test, 46)

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
