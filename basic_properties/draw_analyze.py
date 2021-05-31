import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "CMU Serif",
    "font.size": 15
})

PATH = './output/basic_properties'
GROUPS = [0, 4, 9, 14]
GROUP_NAMES = ["Layer size", "Number layers", "Input order", "Number labels"]


def plot(output_path, data, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #fig.suptitle(title)
    ax.imshow(data)
    ticks = np.arange(data.shape[0]).tolist()
    tick_labels = np.arange(1, data.shape[0]+1).tolist()
    minor_ticks = [group - 0.5 for group in GROUPS[1:]]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
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
    dirs_data = dict()
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if '.npy' in file:
                path = os.path.join(root, file)
                dirs_data[path] = np.load(path)

    stats = list()
    for directory, data in dirs_data.items():
        dir_split = directory.split('\\')[1:]
        dataset = dir_split[0]
        distance_mat_name = dir_split[1].split('_')[0]
        data_std = np.std(data, axis=0)
        data = np.mean(data, axis=0)
        data_norm = data / np.max(data)
        for idx, experiment_idx in enumerate(GROUPS):
            if idx < (len(GROUPS) - 1):
                data_part = data_norm[experiment_idx:GROUPS[idx + 1], experiment_idx:GROUPS[idx + 1]]
            else:
                data_part = data_norm[experiment_idx:, experiment_idx:]
            data_part = data_part[np.triu_indices_from(data_part, k=1)]
            stats.append({'Dataset': dataset, 'Discretization': distance_mat_name, 'Experiment': GROUP_NAMES[idx],
                          'Mean': np.mean(data_part), 'Standard deviation': np.std(data_part)})

        # Plot
        base_title = f'{dataset.replace("_", " ")} - {distance_mat_name.capitalize()}'
        plot(f'{os.path.splitext(directory)[0]}.pdf', data, f'{base_title} - Means')
        plot(f'{os.path.splitext(directory)[0]}_std.pdf', data_std, f'{base_title} - Standard deviations')
    pd.DataFrame(stats).to_csv(os.path.join(PATH, 'stats.csv'), index=False, float_format='%.4f')
