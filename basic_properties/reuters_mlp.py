import os
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from itertools import combinations
from graph import model2graphig
from homology import graphigs2vrs
from gtda.diagrams import PairwiseDistance, Filtering
from basic_properties.conf.conf_reuters import DROPOUT_SEED, ANALYSIS_TYPES
import functools
from collections import defaultdict

# MLP training
batch_size = 128
num_classes = 46
epochs = 20
TIMES = 5
MAXLEN = 100
EXPERIMENT_NAME = "REUTERS"
METRICS = ['silhouette', 'landscape', 'heat']

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


'''
This assumes that all the process involve the same number of random number
petitions.

As an example:

Proc1 execution:
random1()
random2()
initialization()

Proc2 execution:
random1()
initialization()

Proc1 and Proc2 won't have same initialization as the state has been modified
different times.
 
'''
# np.random.seed(SEED)
# RANDOM_STATE = np.random.get_state()


if __name__ == '__main__':
    # Computing
    distances_all = defaultdict(list)
    for _ in range(TIMES):
        graphs = list()
        for analysis in ANALYSIS_TYPES:
            analysis_type, analysis_values = analysis['name'], analysis['values']
            for analysis_value in analysis_values:
                # np.random.seed(SEED)
                # np.random.set_state(RANDOM_STATE)
                # print("Randomization control:", np.random.rand(10))
                model = tf.keras.Sequential()

                if analysis_type == 'LAYER_SIZE':
                    model.add(Dense(analysis_value, activation='relu', input_shape=(10000,),
                                    kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    model.add(Dense(analysis_value, activation='relu', kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                elif analysis_type == 'NUMBER_LAYERS':
                    model.add(
                        Dense(512, activation='relu', input_shape=(10000,), kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    for _ in range(analysis_value - 1):
                        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
                        model.add(Dropout(0.2, seed=DROPOUT_SEED))
                else:
                    model.add(
                        Dense(512, activation='relu', input_shape=(10000,), kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))

                if analysis_type == 'NUMBER_LABELS':
                    model.add(Dense(analysis_value, activation='softmax'))
                else:
                    model.add(Dense(num_classes, activation='softmax'))

                # the data, split between train and test sets
                (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
                # Vectorized training data
                x_train = vectorize_sequences(train_data)
                # Vectorized test data
                x_test = vectorize_sequences(test_data)

                y_train = np.asarray(train_labels).astype('float32')
                y_test = np.asarray(test_labels).astype('float32')
                # convert class vectors to binary class matrices
                if analysis_type == 'NUMBER_LABELS':
                    # Select labels
                    labels = sorted(list(set(y_train)))
                    labels = labels[:analysis_value]

                    # Train filter
                    train_idxs = np.where(np.isin(y_train, labels))
                    x_train = x_train[train_idxs]
                    y_train = y_train[train_idxs]

                    # Test filter
                    test_idxs = np.where(np.isin(y_test, labels))
                    x_test = x_test[test_idxs]
                    y_test = y_test[test_idxs]

                    # Categories
                    y_train = tf.keras.utils.to_categorical(y_train, len(labels))
                    y_test = tf.keras.utils.to_categorical(y_test, len(labels))
                elif analysis_type == 'INPUT_ORDER':
                    # Get indexes
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
                    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
                    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

                    # Reset seed for model initialization. BE CAREFUL.
                    # np.random.seed(SEED)
                    # np.random.set_state(RANDOM_STATE)
                else:
                    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
                    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

                model.summary()
                model.compile(loss='categorical_crossentropy',
                              optimizer=RMSprop(),
                              metrics=['accuracy'])
                history = model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_split=0.1)

                G = model2graphig(model, method='reverse')
                graphs.append(G)

        diagrams = graphigs2vrs(graphs)

        # Filter
        print("Before filtering", diagrams.shape)
        diagrams = Filtering(epsilon=0.01).fit_transform(diagrams)
        print("After filtering", diagrams.shape)

        # Replace
        diagrams[diagrams == np.Inf] = 1.0

        # Compute distances
        for metric in METRICS:
            start = time.time()
            distances = list()
            for idx_0, idx_1 in tqdm(combinations(range(diagrams.shape[0]), r=2)):
                dist = PairwiseDistance(metric=metric, n_jobs=1, metric_params={'n_bins': 200}).fit_transform(
                    np.take(diagrams, [idx_0, idx_1], axis=0))
                distances.append((idx_0, idx_1, dist[0][1]))
            end = time.time()
            print("Elapsed time:", end - start)
            max_n = max([max(distances, key=lambda x: x[0])[0], max(distances, key=lambda x: x[1])[1]]) + 1
            final_data = np.zeros((max_n, max_n))
            for d in distances:
                final_data[d[0]][d[1]] = d[2]
                final_data[d[1]][d[0]] = d[2]

            # Append to distances
            distances_all[metric].append(final_data)

    output_path = os.path.join('./output/basic_properties/', EXPERIMENT_NAME)
    for metric in METRICS:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{metric}_distance_matrices.npy'), 'wb') as f:
            np.save(f, distances_all[metric])
