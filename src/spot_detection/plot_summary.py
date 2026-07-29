
import matplotlib
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from smma.src import visualise
from loguru import logger

logger.info('Import OK')

input_path = 'results/spot_detection/count_spots/spots_per_fov.csv'
output_folder = 'results/spot_detection/plot_summary/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font = {'family': 'normal',
        'weight': 'normal',
        'size': 12}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

# Read in summary FOV data
spots = pd.read_csv(f'{input_path}')
spots.drop([col for col in spots.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# Drop extra wells collected by default due to grid pattern
spots.dropna(subset=['sample'], inplace=True)

# expand sample info
spots[['capture', 'sample', 'detect']
      ] = spots['sample'].str.split('_', expand=True)
spots['spots_count'] = spots['spots_count'].fillna(0)
# spots['spots_count'] = spots['spots_count'].replace(0, np.nan)


# plot average spots per well heatmap
fig, ax = plt.subplots(figsize=(20, 10))
visualise.plot_parameters_heatmap(spots, ax=False, figsize=(
    20, 10), cmap='Purples', capture_col='capture', sample_col='sample', detection_col='detect')
plt.savefig(f'{output_folder}641_heatmap.png')

# Plot summary
spots = spots[spots['sample'] != 'Beads']
spots['color_key'] = [f'{x}_{y}' for x, y in spots[['protein', 'conc']].values]

# Create superplot summary
palette = {
    '': '#0B3C49',
    '': 'lightgrey',
    '': '#0B3C49',
    '': '#CBD2D0',
    '': '#731963',
    '': '#CC7E85',
}

for (detect, protein), df in spots.groupby(['detect', 'protein']):

    fig, ax = plt.subplots()
    sns.stripplot(
        data=df,
        x='timepoint',
        # order=[],
        y='spots_count',
        hue='color_key',
        palette=palette,
        marker="$\circ$", ec="face",
        size=8
    )
    sns.stripplot(
        data=df.groupby(['timepoint', 'conc', 'color_key']).mean().reset_index(),
        x='timepoint',
        # order=[],
        y='spots_count',
        hue='color_key',
        palette=palette,
        s=10,
        linewidth=1,
        edgecolor='white'
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.0, 1.0))
    ax.set_ylabel('Particles per field of view')
    ax.set_xlabel('Time point (h)')
    # ax.set_ylim(0, 500)
    plt.title(f"{protein} {detect}")
    plt.tight_layout()
    plt.savefig(
        f'{output_folder}superplot_{detect}_{protein}.png')
    plt.savefig(
        f'{output_folder}superplot_{detect}_{protein}.svg')

