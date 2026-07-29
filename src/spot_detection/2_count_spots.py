import functools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io
from loguru import logger

from smma.src.utilities import comdet
import smma.src.utilities as utilities

logger.info('Import OK')

# Remember to add napari.run() calls after each loop!
# SETTINGS.application.ipy_interactive = False
np.set_printoptions(suppress=True)  # stop scientific notion for printing

image_id = '14072026_ADcontrolCurve22026-07-14_16-57-57'

input_folder = utilities.locate_raw_drive_files(
    input_path='raw_data/raw_data.txt'
).strip()

image_folder = os.path.join(input_folder, image_id)

input_path = 'results/spot_detection/initial_cleanup/slide_parameters.csv'
output_folder = 'results/spot_detection/count_spots/'

control_image = 'X8Y0R3W2C0_647_641'
test_image = 'X9Y0R3W2C0_647_641'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Read in file directory
slide_params = pd.read_csv(f'{input_path}')
slide_params.drop([col for col in slide_params.columns.tolist()
                  if 'Unnamed: ' in col], axis=1, inplace=True)
# remove discard images
slide_params.dropna(subset=['keep'], inplace=True)

# ----------Determine optimal parameters----------
# Generate projection image - in this case, maximum

max_pixels = 10
max_threshold = 20

control_spots = []

for image_name in [control_image, test_image]:
    image_path = os.path.join(
        image_folder,
        f"{image_name}.tif"
    )

    print("Reading:", image_path)
    print("Exists:", os.path.isfile(image_path))

    image = io.imread(image_path)
    mean_image = np.mean(image, axis=0)

    for pixel_size in range(2, max_pixels + 1):
        for threshold in range(3, max_threshold + 1):
            visualise = False

            measurements = comdet(
                image=mean_image.astype(float),
                sigma_threshold=threshold,
                particle_guess=pixel_size,
                visualise=visualise
            )

            logger.info(
                f"{len(measurements)} spots detected in {image_name} "
                f"with particle guess {pixel_size} and threshold {threshold}"
            )

            measurements["particle_guess"] = pixel_size
            measurements["sigma_threshold"] = threshold
            measurements["image_name"] = image_name
            control_spots.append(measurements)

spots = pd.concat(control_spots)


spots_per_fov = spots.groupby(['particle_guess', 'sigma_threshold', 'image_name']).count(
)['label'].reset_index().rename(columns={'label': 'spots_count'})

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for x, image_name in enumerate([control_image, test_image]):
    sns.lineplot(
        data=spots_per_fov[spots_per_fov['image_name'] == image_name],
        x='sigma_threshold',
        y='spots_count',
        hue='particle_guess',
        ax=axes[x]
        )
    axes[x].set_title(image_name)
plt.tight_layout()
plt.show()

comparison = pd.pivot(spots_per_fov, index=['particle_guess', 'sigma_threshold'], columns=['image_name'], values='spots_count').reset_index()
comparison['proportion'] = comparison[control_image] / comparison[test_image]
fig, ax = plt.subplots()
sns.lineplot(
    data=comparison,
    x='sigma_threshold',
    y='proportion',
    hue='particle_guess',
    palette='tab10'
)
sns.scatterplot(
    data=comparison[comparison['proportion'] < 0.1].sort_values('sigma_threshold').drop_duplicates(subset=['particle_guess']),
    x='sigma_threshold',
    y='proportion',
    hue='particle_guess',
    palette='tab10'
)

plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()

# -----------------localize_spots-----------------
thresholds = {
    '641': [8, 6],
}

slide_details = dict()
# Detect spots
spots = []
for x, (layout, image_path, slide_name) in enumerate(slide_params[['layout', 'file_path', 'well_info']].values):
    channel = slide_name.split('_')[-1]
    sigma_threshold, particle_guess = thresholds[channel]

    # Generate projection image - in this case, maximum
    image = io.imread(image_path)
    mean_image = np.mean(image[10:, :, :], axis=0)

    visualise = f'{output_folder}{slide_name}_spots.png' if x % 16 == 0 else False
    measurements = comdet(image=mean_image.astype(
        float), sigma_threshold=sigma_threshold, particle_guess=particle_guess, visualise=visualise)
    logger.info(f'{len(measurements)} spots detected in {slide_name}')

    measurements['layout'] = layout
    measurements['well_info'] = slide_name
    measurements['file_path'] = image_path

    spots.append(measurements)
spots = pd.concat(spots)

spots_per_fov = spots.groupby(['file_path', 'well_info', 'layout']).count(
)['label'].reset_index().rename(columns={'label': 'spots_count'})


# Add sample info
spots_per_fov = functools.reduce(lambda left, right: pd.merge(
    left, right, on=['file_path', 'well_info', 'layout'], how='outer'), [slide_params, spots_per_fov])
# Save to csv
spots_per_fov.to_csv(f'{output_folder}spots_per_fov.csv')


# Add sample info
spots = functools.reduce(lambda left, right: pd.merge(
    left, right, on=['file_path', 'well_info', 'layout'], how='outer'), [slide_params, spots])
# Save to csv
spots.to_csv(f'{output_folder}compiled_spots.csv')


# Add sample info
spot_stats = spots.groupby(
    ['file_path', 'well_info', 'layout']
).mean(numeric_only=True)
spot_stats = functools.reduce(lambda left, right: pd.merge(
    left, right, on=['file_path', 'well_info', 'layout'], how='outer'), [slide_params, spot_stats])
spot_stats.drop([col for col in spot_stats.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
# Save to csv
spot_stats.to_csv(f'{output_folder}compiled_stats.csv')