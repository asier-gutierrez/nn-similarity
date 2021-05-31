import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "CMU Serif",
    "font.size": 15
})

# from basic_properties.input_order import NAMES as MODEL_NAMES
MODEL_NAMES = ["MNIST", "Fashion MNIST", "CIFAR-10","Reuters", "Language"]

PATH = './output/basic_properties_number_labels'


def plot(output_path, data, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig.suptitle(title)
    ax.imshow(data)
    ticks = [group * 5 + 2.5 for group in range(data.shape[0] // 5)]
    minor_ticks = [group * 5 - 0.5 for group in range(data.shape[0] // 5)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(MODEL_NAMES)
    ax.set_yticks(ticks)
    ax.set_yticklabels(MODEL_NAMES)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(which='minor', length=0)
    # ax.tick_params(axis='both', which='minor', color='r')
    ax.grid(True, which='minor', linestyle='--', color='#696969')
    ax.grid(True, which='minor', linestyle='--', color='w', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')


if __name__ == '__main__':
    output_path = os.path.join(PATH, 'outputs')
    os.makedirs(output_path, exist_ok=True)

    data_path = list()
    auto_distances_mean = dict()
    auto_distances_std = dict()
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if '.npy' in file and 'autodistance' not in file:
                path = os.path.join(root, file)
                data_path.append(path)
            elif '.npy' in file and 'autodistance' in file:
                auto_distance = np.load(os.path.join(root, file))
                distance = '_'.join(file.split('_')[0:1])
                auto_distances_mean[distance] = np.repeat(np.mean(auto_distance, axis=1), 5)
                auto_distances_std[distance] = np.repeat(np.std(auto_distance, axis=1), 5)

    stats = list()
    for path in data_path:
        base_path = os.path.dirname(path)
        distance_mat_name = os.path.basename(path)
        distance = '_'.join(distance_mat_name.split('_')[0:1])
        distance_mat_name = distance_mat_name[:distance_mat_name.rindex('_')]
        distance_mat_name = distance_mat_name.replace('_', ' ').capitalize()

        data = np.load(path)
        not_topological = ("Landscape" not in distance_mat_name) and ("Silhouette" not in distance_mat_name) and (
                "Heat" not in distance_mat_name)
        if not_topological:
            data[data == 0.0] = None

        data_std = np.std(data, axis=0)
        data = np.mean(data, axis=0)
        groups = data.shape[0] // 5
        diagonal_max_mean = list()
        for idx in range(groups):
            if idx < groups:
                d = data[idx * 5:(idx + 1) * 5, idx * 5:(idx + 1) * 5].copy()
            else:
                d = data[idx * 5:, idx * 5:].copy()
            d[np.isnan(d)] = 0.0
            diagonal_max_mean.append(np.max(d))
        if not_topological:
            diagonal_max_mean = np.repeat(diagonal_max_mean, 5)
            diag_differences_mean = list(
                map(lambda x: abs(x[1] - x[0]) / x[0], zip(diagonal_max_mean, auto_distances_mean[distance])))
            print(f'{distance},{np.min(diag_differences_mean):.4f},'
                  f'{np.max(diag_differences_mean):.4f},'
                  f'{np.mean(diag_differences_mean):.4f},'
                  f'{np.std(diag_differences_mean):.4f}')

        data_norm = data / np.max(data)
        for idx in range(groups):
            if idx < (groups - 1):
                data_part = data_norm[idx * 5:(idx + 1) * 5, idx * 5:(idx + 1) * 5]
            else:
                data_part = data_norm[idx * 5:, idx * 5:]
            data_part = data_part[np.triu_indices_from(data_part, k=1)]
            stats.append({'Distance': distance_mat_name, 'Experiment': MODEL_NAMES[idx],
                          'Mean': np.mean(data_part), 'Standard deviation': np.std(data_part)})

        # Plot
        base_title = f'{distance_mat_name}'
        plot(os.path.join(output_path, f'{base_title}.pdf'), data, f'{base_title} - Means')
        plot(os.path.join(output_path, f'{base_title}_std.pdf'), data_std, f'{base_title} - Standard deviations')
    pd.DataFrame(stats).to_csv(os.path.join(output_path, 'stats.csv'), index=False, float_format='%.4f')
