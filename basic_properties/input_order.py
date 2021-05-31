import os
import time
import numpy as np
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from gtda.diagrams import Filtering, PairwiseDistance
from homology import graphigs2vrs
from graph import model2graphig
from basic_properties.input_order_models.mnist_fashion_mlp import prepare as prepare_mnist_fashion, \
    train as train_mnist_fashion
from basic_properties.input_order_models.cifar_mlp import prepare as prepare_cifar, \
    train as train_cifar
from basic_properties.input_order_models.mnist_mlp import train as train_mnist
from basic_properties.input_order_models.reuters_mlp import train as reuters_train
from basic_properties.input_order_models.language_identification import prepare as prepare_language_identification, \
    train as train_language_identification


def distances_norm(graphs, norm):
    start = time.time()
    distances = list()
    for i in np.arange(0, len(graphs), 5):
        for idx_0, idx_1 in tqdm(combinations(range(RUNS), r=2)):
            dist = np.linalg.norm(np.array((graphs[idx_0+i].get_adjacency(attribute='weight') - graphs[idx_1+i].get_adjacency(attribute='weight')).data), norm)
            distances.append((idx_0+i, idx_1+i, dist))
    end = time.time()
    print("Elapsed time:", end - start)
    max_n = max([max(distances, key=lambda x: x[0])[0], max(distances, key=lambda x: x[1])[1]]) + 1
    final_data = np.zeros((max_n, max_n))
    for d in distances:
        final_data[d[0]][d[1]] = d[2]
        final_data[d[1]][d[0]] = d[2]

    # Append to distances
    return final_data


def auto_distances_norm(graphs, norm):
    return [np.linalg.norm(np.array(g.get_adjacency(attribute='weight').data), norm) for g in graphs]

EXECUTIONS = 5
RUNS = 5
METRICS = ['silhouette', 'landscape', 'heat']
NAMES = ["MNIST", "Fashion MNIST", "CIFAR-10", "Reuters", "Language Identification"]
if __name__ == '__main__':
    # Load tl models
    mnist_fashion_model = prepare_mnist_fashion()
    cifar_model = prepare_cifar()
    language_identification_data = prepare_language_identification()

    distances_all = defaultdict(list)
    distances_fro = list()
    auto_distances_fro = list()
    distances_1 = list()
    auto_distances_1 = list()
    for _ in range(EXECUTIONS):
        graphs = list()
        for _ in range(RUNS):
            graphs.append(model2graphig(train_mnist(), method='reverse'))
        for _ in range(RUNS):
            graphs.append(model2graphig(train_mnist_fashion(mnist_fashion_model), method='reverse'))
        for _ in range(RUNS):
            graphs.append(model2graphig(train_cifar(cifar_model), method='reverse'))
        for _ in range(RUNS):
            graphs.append(model2graphig(reuters_train(), method='reverse'))
        for _ in range(RUNS):
            graphs.append(model2graphig(train_language_identification(*language_identification_data), method='reverse'))

        # Frobenius norm
        distances_fro.append(distances_norm(graphs, 'fro'))
        auto_distances_fro.append(auto_distances_norm(graphs, 'fro'))

        # 1-norm
        distances_1.append(distances_norm(graphs, 1))
        auto_distances_1.append(auto_distances_norm(graphs, 1))

        # PH
        diagrams = graphigs2vrs(graphs)

        # Filter diagrams
        print("Before filtering", diagrams.shape)
        diagrams = Filtering(epsilon=0.01).fit_transform(diagrams)
        print("After filtering", diagrams.shape)
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

    output_path = os.path.join('./output/basic_properties_number_labels/')
    for metric in METRICS:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f'{metric}_distance_matrices.npy'), 'wb') as f:
            np.save(f, distances_all[metric])
    with open(os.path.join(output_path, f'frobenius_distance_matrices.npy'), 'wb') as f:
        np.save(f, distances_fro)
    with open(os.path.join(output_path, f'1_distance_matrices.npy'), 'wb') as f:
        np.save(f, distances_1)
    with open(os.path.join(output_path, f'frobenius_autodistance_matrices.npy'), 'wb') as f:
        np.save(f, auto_distances_fro)
    with open(os.path.join(output_path, f'1_autodistance_matrices.npy'), 'wb') as f:
        np.save(f, auto_distances_1)
