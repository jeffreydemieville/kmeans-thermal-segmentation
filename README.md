# kmeans-thermal-segmentation
This script extracts temperatures of the plant canopy and soil from thermal imagery. Order of operations includes:
1. Read in all pixel values from all images, discarding NaN and downsampling if necessary
2. Use K-means clustering and separate the data into two clusters. Extract the cluster centers, select pixels between the centers, then compute the histogram and find local minimum.
3. Define plant temp as values below the threshold and soil temp as values above the threshold. Return mean values for each, difference between the means, and percentiles for plants pixels.
4. Save results of global threshold calculation.
5. For each image, use the threshold calculated to individually threshold each image.
6. For each image, extract lat/lon.
7. Save results of extraction.
8. Randomly select 5 images and display original and thresholded pixels on the same scale. Save to PNG.

## Arguments
* dir: required, directory containing input geotiffs
* -d/--date: required, used for naming outputs and including scan date in output CSVs
* -od/--outdir: optional, used to set directory containing outputs, default='temps_out'
* -g/--global_name: optional, used to set output filename component for global thresholding results, default='global'
* -i/--individual_name: optional, used to set output filename component for individual thresholding results, default='individual'

![Sample thresholding](/sample_thresholded_comparison_images.png)

## Prerequisites
Extract thermal geotiff images to a known location. Script expects a directory containing tif images with a header preceding an underscore. The header is captured in the output data for each row.
E.g.
```
plotclip_orthos/
plotclip_orthos/0144_ortho.tif
plotclip_orthos/0204_ortho.tif
plotclip_orthos/0203_ortho.tif
plotclip_orthos/0202_ortho.tif
plotclip_orthos/0201_ortho.tif
plotclip_orthos/0146_ortho.tif
```

## Building the Docker container
```
docker build -t kmeans-thermal-segmentation .
```

## Running the Docker container
```
docker run -v "$(pwd):/app" kmeans-thermal-segmentation plotclip_orthos -d 2025-07-21__17-29-15-000_cotton -od plot_temps_out -g global -i plot
```
This assumes that plot-level clipped orthomosaic thermal imagery captured at 2025-07-21 17-29-15-000 on cotton is extracted to a directory plotclip_orthos. Outputs are stored in a directory plot_temps_out. Global thresholds are stored in ./plot_temps_out/2025-07-21__17-29-15-000_cotton_global_thresholding_results.csv. Plot-level results are stored in ./plot_temps_out/2025-07-21__17-29-15-000_cotton_plot_thresholding_results.csv. An image for quality control is saved in ./plot_temps_out/2025-07-21__17-29-15-000_cotton_thresholded_comparison_images.png.
