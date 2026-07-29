import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from smma.src import stat_tests as statistics
from smma.src import utilities

from loguru import logger
logger.info('Import OK')

input_folder = 'results/spot_detection/count_spots/'
output_folder = 'results/spot_detection/comparison/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# import summary info
spots = pd.read_csv(f'{input_folder}spots_per_fov.csv')
spots.drop([col for col in spots.columns.tolist()
            if 'Unnamed: ' in col], axis=1, inplace=True)
# fill slides with no spots
spots['spots_count'] = spots['spots_count'].fillna(0)


# At a single threshold, test for differences in each sample type
anova_spots = utilities.prepare_spots_for_anova(
    spots.rename(columns={'spots_count': 'count_spots_per_image'}), output_folder=False)
anova_spots.dropna(how='all')
anova_spots.columns = [f'{i}_{j}' if j !=
                       '' else f'{i}' for i, j in anova_spots.columns]
anova_spots.to_csv(f'{output_folder}anova_spots.csv')

# Compare samples via ANOVA
anova_spots.columns.tolist()
sample_comparisons = [
    anova_spots.columns.tolist(),
]

anova, multi_comp = statistics.apply_anova_tukey(
    anova_spots, sample_comparisons)

# Save summary statistics
anova.to_csv(f'{output_folder}anova.csv')
multi_comp.to_csv(f'{output_folder}tukeys_multiple_comparison.csv')
