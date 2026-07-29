import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sns
import random
from smma.src import visualise, utilities

from loguru import logger

logger.info('Import OK')

image_id = ''
input_folder = utilities.locate_raw_drive_files(
    input_path='raw_data/raw_data.txt')
image_folder = f'{input_folder}/{image_id}/'
input_spots = 'results/spot_detection/count_spots/compiled_spots.csv'
input_params = 'results/spot_detection/initial_cleanup/slide_parameters.csv'

output_folder = 'results/spot_detection/plot_examples/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Read in spot data
compiled_spots = pd.read_csv(f'{input_spots}')
compiled_spots.drop([col for col in compiled_spots.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
compiled_spots.rename( columns={'centroid-1': 'x', 'centroid-0': 'y'}, inplace=True)
# Read in slide details
slide_params = pd.read_csv(f'{input_params}')
slide_params.drop([col for col in slide_params.columns.tolist()
                  if 'Unnamed: ' in col], axis=1, inplace=True)

# plot summary image to visualise highlighted spots
visualise.image_spots_overview(
    slide_params[slide_params['channel'] == 641], compiled_spots, figsize=(20, 40), output_folder=output_folder)

# Plot individual sample image
example_wells = random.sample([slide for slide in slide_params.dropna()[
    'well_info'].tolist()], 40)

visualise.image_spots(
    example_wells, slide_params, compiled_spots, output_folder=output_folder)
