import os
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from itertools import combinations
from graph import model2graphig
from homology import graphigs2vrs
from gtda.diagrams import PairwiseDistance, Filtering
from basic_properties.conf.conf_mnist import DROPOUT_SEED, ANALYSIS_TYPES
import functools
from collections import defaultdict

# MLP training
batch_size = 128
num_classes = 10
epochs = 20
TIMES = 5
EXPERIMENT_NAME = "MNIST"
METRICS = ['silhouette', 'landscape', 'heat']

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
                    model.add(Dense(analysis_value, activation='relu', input_shape=(784,), kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    model.add(Dense(analysis_value, activation='relu', kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                elif analysis_type == 'NUMBER_LAYERS':
                    model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    for _ in range(analysis_value - 1):
                        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
                        model.add(Dropout(0.2, seed=DROPOUT_SEED))
                else:
                    model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))
                    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
                    model.add(Dropout(0.2, seed=DROPOUT_SEED))

                if analysis_type == 'NUMBER_LABELS':
                    model.add(Dense(analysis_value, activation='softmax'))
                else:
                    model.add(Dense(num_classes, activation='softmax'))


                # the data, split between train and test sets
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
                    #np.random.seed(analysis_value)
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
                    #np.random.seed(SEED)
                    #np.random.set_state(RANDOM_STATE)
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
        '''
        def filter(diagrams):
            subtractions = list()
            for diagram in diagrams:
                for bdq in diagram:
                    if not np.isinf(bdq[1]):
                        subtractions.append(bdq[1] - bdq[0])
                    else:
                        subtractions.append(1 - bdq[0])
    
            subs_mean = np.mean(subtractions)
            subs_std = np.std(subtractions)
            print(subs_mean, subs_std)
            return subs_mean, subs_std
    
        subs_mean, subs_std = filter(diagrams)
        '''
        print("Before filtering", diagrams.shape)
        diagrams = Filtering(epsilon=0.01).fit_transform(diagrams)
        print("After filtering", diagrams.shape)

        # Filter
        '''
        subs_mean, subs_std = filter(diagrams)
        '''
        # Filter infinite numbers and replace with max_n
        '''
        diagrams_cp = np.copy(diagrams)
        diagrams_cp[:, :, 2] = 0.0
        diagrams_cp = diagrams_cp.flatten()
        diagrams_cp = diagrams_cp[np.where(diagrams_cp != np.Inf)]
        max_n = max(diagrams_cp)
        '''

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
