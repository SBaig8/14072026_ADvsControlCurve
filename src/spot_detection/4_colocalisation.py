import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functools

from smma.src.utilities import closest_node, find_pairs
from smma.src.visualise import plot_colocalisation

from loguru import logger

logger.info('Import OK')

input_path = 'results/spot_detection/count_spots/compiled_spots.csv'
input_parameters = 'results/spot_detection/initial_cleanup/slide_parameters.csv'
output_folder = 'results/spot_detection/colocalisation/'

channel_1 = '638'
channel_2 = '488'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read in spots
spots = pd.read_csv(input_path)
spots.drop([col for col in spots.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
spots['channel'] = spots['well_info'].str.split('_').str[-3]
spots['fov'] = spots['well_info'].str.split('_').str[0]

# Collect single test image determine threshold 
test_spots = spots[(spots['fov'] == 'X6Y1R1W1C0') & (spots['layout'] == '2a')].copy()
for threshold in [2, 3, 4, 5]:
    logger.info(f'Threshold: {threshold}')
    plot_colocalisation(test_spots[test_spots['channel'] == channel_1], test_spots[test_spots['channel'] == channel_2], threshold=threshold)

# Determined threshold
threshold = 3

# -------------Apply colocalisation to all images-------------
colocalised = []
for (well_info, layout), df in spots.groupby(['fov', 'layout']):
    seed_spots = df[df['channel'] == channel_1].copy()
    test_spots = df[df['channel'] == channel_2].copy()
    if any(len(df) < 1 for df in [seed_spots, test_spots]):
        continue

    pairs = find_pairs(
        seed_spots[['centroid-0', 'centroid-1']].values, 
        test_spots[['centroid-0', 'centroid-1']].values,
        threshold=threshold
    ).dropna(subset=['test_coord_0'])
    pairs['spot_id'] = np.arange(1, len(pairs)+1)

    # visualise_colocalisation(df[df['channel'] == channel_2], df[df['channel'] == channel_1], threshold=threshold)
    pair_ids = dict(zip([tuple(x) for x in pairs[['seed_coord_0', 'seed_coord_1']].values], pairs['spot_id'].tolist()))
    pair_ids.update(dict(zip([tuple(x) for x in pairs[['test_coord_0', 'test_coord_1']].values], pairs['spot_id'].tolist())))

    seed_spots['pair_id'] = [pair_ids[coord] if coord in pair_ids else np.nan for coord in [tuple(x) for x in seed_spots[['centroid-0', 'centroid-1']].values]]
    test_spots['pair_id'] = [pair_ids[coord] if coord in pair_ids else np.nan for coord in [tuple(x) for x in test_spots[['centroid-0', 'centroid-1']].values]]

    colocalised.extend([seed_spots, test_spots])
colocalised = pd.concat(colocalised)

colocalised['coloc?'] = [0 if np.isnan(id) else 1 for id in colocalised['pair_id']]

# -------------Apply colocalisation to transposed images-------------
randomised = []
for (well_info, layout), df in spots.groupby(['fov', 'layout']):
    seed_spots = df[df['channel'] == channel_1].copy()
    test_spots = df[df['channel'] == channel_2].copy()
    test_spots['centroid-rev'] = 512.0 - test_spots['centroid-1']
    
    if any(len(df) < 1 for df in [seed_spots, test_spots]):
        continue

    pairs = find_pairs(
        seed_spots[['centroid-0', 'centroid-1']].values, 
        test_spots[['centroid-rev', 'centroid-0']].values,
        threshold=threshold
    ).dropna(subset=['test_coord_0'])
    pairs['spot_id'] = np.arange(1, len(pairs)+1)

    # visualise_colocalisation(df[df['channel'] == channel_2], df[df['channel'] == channel_1], threshold=threshold)
    pair_ids = dict(zip([tuple(x) for x in pairs[['seed_coord_0', 'seed_coord_1']].values], pairs['spot_id'].tolist()))
    pair_ids.update(dict(zip([tuple(x) for x in pairs[['test_coord_0', 'test_coord_1']].values], pairs['spot_id'].tolist())))

    seed_spots['pair_id'] = [pair_ids[coord] if coord in pair_ids else np.nan for coord in [tuple(x) for x in seed_spots[['centroid-0', 'centroid-1']].values]]
    test_spots['pair_id'] = [pair_ids[coord] if coord in pair_ids else np.nan for coord in [tuple(x) for x in test_spots[['centroid-rev', 'centroid-0']].values]]

    randomised.extend([seed_spots, test_spots])
randomised = pd.concat(randomised)

randomised['coloc?'] = [0 if np.isnan(id) else 1 for id in randomised['pair_id']]

# ------------------Calculate # colocalised in each position------------------
coloc_summary = []
for (layout, slide_pos), df in colocalised.groupby(['layout', 'fov']):

    total_spots = df.groupby('channel').count()['well_info']
    coloc_spots = df.groupby('channel').count()['pair_id']
    proportion_colocalised = (coloc_spots / total_spots * 100).reset_index().rename(columns={0: 'proportion_coloc'})

    total_spots = total_spots.reset_index().rename(columns={'well_info': 'total_spots'})
    coloc_spots = coloc_spots.reset_index().rename(columns={'pair_id': 'coloc_spots'})

    mean_intensity = df.groupby(['channel', 'coloc?']).mean()['mean_intensity'].reset_index()
    mean_intensity = pd.pivot_table(mean_intensity, values='mean_intensity', index='channel', columns='coloc?')
    mean_intensity.columns = ['mean_intensity-coloc' if x == 1 else 'mean_intensity-noncoloc' for x in mean_intensity.columns]

    # Determine randomised colocalisation
    random_colocalisation = randomised[(randomised['layout'] == layout) & (randomised['fov'] == slide_pos)].copy()
    rand_coloc = random_colocalisation.groupby('channel').count()['pair_id']
    rand_coloc_proportion = (rand_coloc / total_spots * 100).reset_index().rename(columns={0: 'proportion_randcoloc'})

    rand_coloc = rand_coloc.reset_index().rename(columns={'pair_id': 'randcoloc_spots'})

    merged_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['channel'], how='outer'), [total_spots, coloc_spots, proportion_colocalised, mean_intensity])

    merged_df['fov'] = slide_pos
    merged_df['layout'] = layout

    coloc_summary.append(merged_df)
coloc_summary = pd.concat(coloc_summary)

# ------------Generate randomised summary------------
chance_summary = []
for (layout, slide_pos), df in randomised.groupby(['layout', 'fov']):
    total_spots = df.groupby('channel').count()['well_info']
    coloc_spots = df.groupby('channel').count()['pair_id']
    proportion_colocalised = (coloc_spots / total_spots * 100).reset_index().rename(columns={0: 'chance_proportion_coloc'})

    total_spots = total_spots.reset_index().rename(columns={'well_info': 'chance_total_spots'})
    coloc_spots = coloc_spots.reset_index().rename(columns={'pair_id': 'chance_coloc_spots'})
    
    merged_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['channel'], how='outer'), [total_spots, coloc_spots, proportion_colocalised])
    merged_df['fov'] = slide_pos
    merged_df['layout'] = layout
    
    chance_summary.append(merged_df)
chance_summary = pd.concat(chance_summary)


# Add sample information
parameters = pd.read_csv(f'{input_parameters}')
parameters.drop([col for col in parameters.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
parameters['fov'] = parameters['well_info'].str.split('_').str[0]
parameters['channel'] = parameters['channel'].astype(str)

summary = pd.merge(coloc_summary, parameters[['layout', 'fov', 'sample', 'channel']], on=['layout', 'fov', 'channel'], how='left')
summary = pd.merge(summary, chance_summary[['layout', 'fov', 'channel', 'chance_proportion_coloc']], on=['layout', 'fov', 'channel'], how='left')
summary[['capture', 'sample', 'detect']] = summary['sample'].str.split('_', expand=True)

colocalised_summary = pd.merge(colocalised, parameters[['layout', 'fov', 'sample', 'channel']], on=['layout', 'fov', 'channel'], how='left')
colocalised_summary[['capture', 'sample', 'detect']] = colocalised_summary['sample'].str.split('_', expand=True)


summary.to_csv(f'{output_folder}colocalisation_summary.csv')
colocalised.to_csv(f'{output_folder}colocalisation_spots.csv')