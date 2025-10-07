 #!/usr/bin/env python3
"""
Author : Jeffrey Demieville, Sebastian Calleja
Date   : 2025-10-07
Purpose: Plot-level temperature extraction
"""

import argparse
import os
import numpy as np
import cv2
import tifffile as tifi
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform
import random

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
print("libraries imported")

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Plot-level temperature extraction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        metavar='dir',
                        help='Directory containing geoTIFFs')

    parser.add_argument('-d',
                        '--date',
                        help='Scan date',
                        metavar='date',
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument('-od',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='plot_temps_out')

    return parser.parse_args()


# --------------------------------------------------
def get_latlon_bounds(image_path):
    with rasterio.open(image_path) as src:
        # Get image dimensions
        width, height = src.width, src.height

        # Get CRS and transform
        crs = src.crs
        transform_affine = src.transform

        # Define pixel coordinates for corners and center
        pixel_coords = [
            (0, 0),               # top-left
            (width, 0),           # top-right
            (0, height),          # bottom-left
            (width, height),      # bottom-right
            (width // 2, height // 2)  # center
        ]

        # Convert pixel coordinates to spatial coordinates
        spatial_coords = [transform_affine * (x, y) for x, y in pixel_coords]

        # Transform to lat/lon if needed
        lon, lat = zip(*spatial_coords)
        latlon_coords = transform(crs, 'EPSG:4326', lon, lat)

        return {
            'lon' : latlon_coords[0][4],
            'lat' : latlon_coords[1][4],
            'nw_lat' : latlon_coords[1][0],
            'nw_lon' : latlon_coords[0][0],
            'se_lat' : latlon_coords[1][3],
            'se_lon' : latlon_coords[0][3],
        }


# --------------------------------------------------
def main():
    """Extract plot temperatures"""
    
    print("processing started")
    args = get_args()
    
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    # Set the directory containing the .tif files
    directory = args.dir

    # Get list of all .tif files
    tif_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
    #print(tif_files)
    # Collect all pixel values from all images
    all_pixels = []

    for img_path in tif_files:
        print("reading ", img_path)
        tif_img = tifi.imread(img_path)
        tif_img[tif_img == -10000.00] = np.nan
        tif_img[tif_img == 0.0] = np.nan
        img = tif_img[~np.isnan(tif_img)]
        all_pixels.extend(img.flatten().tolist())

    # Convert to numpy array and reshape for k-means
    print("converting array")
    pixel_vals = np.array(all_pixels, dtype=np.float32).reshape(-1, 1)

    # Downsample and check the histogram again
    print("number of pixels: ", len(pixel_vals))
    sample_size = 1000000
    if len(pixel_vals) > sample_size:
        pixel_vals_sample = pixel_vals[np.random.choice(len(pixel_vals), sample_size, replace=False)]
    else:
        pixel_vals_sample = pixel_vals
    print("Downsampling completed")

    # Apply k-means clustering to global pixel data
    print("applying clustering")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = 2
    _, labels, centers = cv2.kmeans(pixel_vals_sample, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Determine min and max cluster centers
    min_val, max_val = sorted(centers.flatten())

    # Filter pixel values between min and max for histogram
    array_mod = pixel_vals_sample[(pixel_vals_sample > min_val) & (pixel_vals_sample < max_val)]

    # Compute histogram and find local minimum
    counts, bins = np.histogram(array_mod, bins=300)
    threshold_minima = bins[np.argmin(counts)]

    # Compute plant and soil temperature statistics
    plant_pixels = pixel_vals_sample[pixel_vals_sample < threshold_minima]
    soil_pixels = pixel_vals_sample[pixel_vals_sample > threshold_minima]

    plant_temp = np.mean(plant_pixels)
    soil_temp = np.mean(soil_pixels)
    temp_diff = soil_temp - plant_temp
    plant_temp_10 = np.percentile(plant_pixels, 10)
    plant_temp_33 = np.percentile(plant_pixels, 33)

    # Store results in a dictionary and convert to DataFrame
    results = {
        'date': args.date,
        'global_threshold_value': threshold_minima,
        'global_plant_temp': plant_temp,
        'global_soil_temp': soil_temp,
        'global_temp_diff': temp_diff,
        'global_plant_temp_10': plant_temp_10,
        'global_plant_temp_33': plant_temp_33
    }

    df = pd.DataFrame([results])
    df.to_csv(os.path.join(args.outdir, args.date + '_global_thresholding_results.csv'), index=False)
    print("Saved " + os.path.join(args.outdir, args.date + '_global_thresholding_results.png'))

    results_list = []

    for img_path in tif_files:
        # Get Plot Name
        plot_name = img_path.split('/')[-1].split('_')[0]

        # Get Temperature Values
        tif_img = tifi.imread(img_path)
        tif_img[tif_img == -10000.00] = np.nan
        tif_img[tif_img == 0.0] = np.nan
        img = tif_img[~np.isnan(tif_img)]
        plot_pixels = img.flatten().tolist()

        pixel_vals = np.array(plot_pixels, dtype=np.float32).reshape(-1, 1)

        plant_pixels = pixel_vals[pixel_vals < threshold_minima]
        soil_pixels = pixel_vals[pixel_vals > threshold_minima]

        # Get Location Information
        coords = get_latlon_bounds(img_path)
        
        # Output results
        results = {
            'date': args.date,
            'plot': plot_name,
            'lon' : coords['lon'],
            'lat' : coords['lat'],
            'nw_lat' : coords['nw_lat'],
            'nw_lon' : coords['nw_lon'],
            'se_lat' : coords['se_lat'],
            'se_lon' : coords['se_lon'],
            'plot_threshold_value': threshold_minima,
            'plot_plant_temp': np.mean(plant_pixels),
            'plot_soil_temp': np.mean(soil_pixels),
            'plot_temp_diff': np.mean(soil_pixels) - np.mean(plant_pixels),
            'plot_plant_temp_10': np.percentile(plant_pixels, 10),
            'plot_plant_temp_33': np.percentile(plant_pixels, 33)
        }

        results_list.append(results)

    # Save all results to a single CSV
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(args.outdir, args.date + '_all_plot_thresholding_results.csv'), index=False)
    print("Saved " + os.path.join(args.outdir, args.date + '_all_plot_thresholding_results.png'))

    # Randomly select up to 5 images
    selected_files = random.sample(tif_files, min(5, len(tif_files)))

    # Create subplots: 2 rows per image (original and thresholded)
    fig, axes = plt.subplots(nrows=2, ncols=len(selected_files), figsize=(4 * len(selected_files), 8))

    # Plot each selected image and its thresholded version
    for i, img_path in enumerate(selected_files):
        # Read image and apply thresholding
        tif_img = tifi.imread(img_path)
        tif_img[tif_img == -10000.0] = np.nan
        tif_img[tif_img == 0.0] = np.nan
        masked_img = np.where(tif_img < threshold_minima, tif_img, np.nan)

        # Determine common color scale limits
        vmin = np.nanmin(tif_img)
        vmax = np.nanmax(tif_img)

        # Original image
        im1 = axes[0, i].imshow(tif_img, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"Original\n{os.path.basename(img_path)}", fontsize=8)
        axes[0, i].axis('off')
        cbar1 = fig.colorbar(im1, ax=axes[0, i], shrink=0.6)
        cbar1.ax.tick_params(labelsize=6)

        # Thresholded image
        im2 = axes[1, i].imshow(masked_img, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"Thresholded\n{os.path.basename(img_path)}", fontsize=8)
        axes[1, i].axis('off')
        cbar2 = fig.colorbar(im2, ax=axes[1, i], shrink=0.6)
        cbar2.ax.tick_params(labelsize=6)

    # Adjust layout and save the figure
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(os.path.join(args.outdir, args.date + '_thresholded_comparison_images.png'), dpi=300)
    plt.close()
    print("Saved " + os.path.join(args.outdir, args.date + '_thresholded_comparison_images.png'))

    print(f'Done, see outputs in {args.outdir}.')


# --------------------------------------------------
if __name__ == '__main__':
    main()