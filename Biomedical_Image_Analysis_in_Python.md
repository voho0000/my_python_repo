# Biomedical Image Analysis in Python

###### tags: `Python` `Biomedical` `image` 
[TOC]
[有目錄版HackMD筆記](https://hackmd.io/5WLtVYG5SUG_v7e-xKWMoA?both)

Resource : [Biomedical Image Analysis in Python](https://www.datacamp.com/courses/biomedical-image-analysis-in-python)

Exploration
-
### Load images
```python
# Import ImageIO
import imageio
# Load "chest-220.dcm"
im = imageio.imread("chest-220.dcm")
# Print image attributes
print('Image type:', type(im))
print('Shape of image array:', im.shape)
```

### Metadata
Patient demographics and Acquisition information
```python
# Import ImageIO
import imageio
im = imageio.imread('chest-220.dcm')
# Print the available metadata fields
print(im.meta.keys())
```

### Plot images
- cmap controls the color mappings for each value. The "gray" colormap is common, but many others are available.
- vmin and vmax control the color contrast between values. Changing these can reduce the influence of extreme values.
- plt.axis('off') removes axis and tick labels from the image.
```python
# Import ImageIO and PyPlot 
import imageio
import matplotlib.pyplot as plt

# Read in "chest-220.dcm"
im = imageio.imread("chest-220.dcm")

# Draw the image with greater contrast
plt.imshow(im,cmap="gray",vmin=-200,vmax=200)

# Render the image
plt.axis('off')
plt.show()
```

### Stack images
shape: vol[plane, row, col]
```python
# Import ImageIO and NumPy
import imageio
import numpy as np

# Read in each 2D image
im1 = imageio.imread('chest-220.dcm')
im2 = imageio.imread('chest-221.dcm')
im3 = imageio.imread('chest-222.dcm')

# Stack images into a volume
vol = np.stack([im1,im2,im3])
print('Volume dimensions:', vol.shape)
```

### Load volumes
```python
# Import ImageIO
import imageio

# Load the "tcia-chest-ct" directory
vol = imageio.volread( "tcia-chest-ct")

# Print image attributes
print('Available metadata:',vol.meta.keys())
print('Shape of image array:', vol.shape)
```

### Generate subplots
```python
# Import PyPlot
import matplotlib.pyplot as plt

# Initialize figure and axes grid
fig, axes = plt.subplots(nrows=2, ncols=1)

# Draw an image on each subplot
axes[0].imshow(im1, cmap='gray')
axes[1].imshow(im2, cmap='gray')

# Remove ticks/labels and render
axes[0].axis('off')
axes[1].axis('off')
plt.show()
```

### Slice 3D images
Plot every 40th slice of vol in grayscale
```python
# Plot the images on a subplots array 
fig, axes = plt.subplots(nrows=1,ncols=4)

# Loop through subplots and draw image
for ii in range(4):
    im = vol[ii*40,:,:]
    axes[ii].imshow(im,cmap="gray")
    axes[ii].axis('off')
    
# Render the figure
plt.show()
```

### Plot other views
asp是算pixal的長寬比，可從meta中的'sampling'欄位取得數值
```python
# Select frame from "vol"
im1 = vol[:, 256, :]
im2 = vol[:, :, 256]

# Compute aspect ratios
d0, d1, d2 =vol.meta['sampling']
asp1 = d0 / d2
asp2 = d0 / d1

# Plot the images on a subplots array 
fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].imshow(im1, cmap='gray', aspect=asp1)
axes[1].imshow(im2, cmap='gray', aspect=asp1)
plt.show()
```

## Masks and Filters






masked data
```python
import scipy.ndimage as ndi
hist = ndi.histogram(im, min=0, 
                         max=255,
                         bins=256)
```    

- Equalization
    - Distributions often skewed toward low intensities (background values).
    - Equalization: redistribute values to optimize full intensity range.
    - Cumulative distribution function: (CDF) shows proportion of pixels in range.
    - ![](https://i.imgur.com/Pc17UvI.png)

### Intensity
Add a colorbar using plt.colorbar(), then render the plot using the custom function format_and_render_plot()
```python
# Load the hand radiograph
im = imageio.imread("hand-xray.jpg")
print('Data type:', im.dtype)
print('Min. value:',im.min())
print('Max value:', im.max())

# Plot the grayscale image
plt.imshow(im,vmin=0,vmax=255)
plt.colorbar()
format_and_render_plot()
```

### Histogram
Histograms: count number of pixels at each intensity value.
Implemented in scipy.ndimage
higher-dimensional arrays, mask
```python
# Import SciPy's "ndimage" module
import scipy.ndimage as ndi

# Create a histogram, binned at each possible value
hist = ndi.histogram(im,min=0, 
                        max=255,
                        bins=256)

# Create a cumulative distribution function
cdf = hist.cumsum() / hist.sum()

# Plot the histogram and CDF
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(hist, label='Histogram')
axes[1].plot(cdf, label='CDF')
format_and_render_plot()
```

### Masks
- Creating masks:
    - Logical operations result in True / False at each pixel

|Operation| Example|
|-|-|
|Greater	|im > 0|
|Equal to|	im == 1|
|X and Y|	(im > 0) & (im < 5)|
|X or Y|	(im > 10) | (im < 5)|

```python
mask1=im>32
mask2=im>64
#mask in mask1 not in mask2
mask3 = mask1 & ~mask2
```

```python
# Create skin and bone masks
mask_bone = im>=145
mask_skin = (im>=45)&(im<145)

# Plot the masks
fig, axes = plt.subplots(1,2)
axes[0].imshow(mask_skin,cmap='gray')
axes[1].imshow(mask_bone,cmap='gray')
format_and_render_plot()
```

- apply a mask
np.where(condition,x,y)
條件TRUE的話回報值為x，否的話為y
```python
import numpy as np
#或者 im_bone = np.where(im > 64, 1, 0)
im_bone = np.where(im > 64, im, 0)
 plt.imshow(im_bone, cmap='gray')
plt.axis('off')
plt.show()
```
```python
# Import SciPy's "ndimage" module
import scipy.ndimage as ndi

# Screen out non-bone pixels from "im"
mask_bone = im >= 145
im_bone = np.where(mask_bone, im, 0)

# Get the histogram of bone intensities
hist = ndi.histogram(im_bone,1,255,255)

# Plot masked image and histogram
fig, axes = plt.subplots(2,1)
axes[0].imshow(im_bone,cmap='gray')
axes[1].plot(hist)
format_and_render_plot()
```


- Tuning Mask
solve noise的問題
![](https://i.imgur.com/37sS77r.png)
binary_dilation: Add pixels along edges
binary_erosion: Remove pixels along edges
binary_opening: Erode then dilate, "opening" areas near edges
binary_closing: Dilate then erode, "filling in" holes

```python
# Create and tune bone mask
mask_bone = im>=145
mask_dilate = ndi.binary_dilation(mask_bone, iterations=5)
mask_closed = ndi.binary_closing(mask_bone, iterations=5)

# Plot masked images
fig, axes = plt.subplots(1,3)
axes[0].imshow(mask_bone)
axes[1].imshow(mask_dilate)
axes[2].imshow(mask_closed)
format_and_render_plot()
```

### Filter
![](https://i.imgur.com/ZgEsG4o.png) 

- Sharpen:the opposite of smooth
    - Convolution with a sharpening filter:define a set of filter weights, so called a kernal
    -![](https://i.imgur.com/J5YDYQy.png)


scipy.ndimage.filters includes:

median_filter()
uniform_filter()
maximum_filter()
percentile_filter()


```python
# Set filter weights
weights = [[0.11, 0.11, 0.11],
           [0.11, 0.11, 0.11], 
           [0.11, 0.11, 0.11]]

# Convolve the image with the filter
im_filt = ndi.convolve(im, weights)

# Plot the images
fig, axes = plt.subplots(1,2)
axes[0].imshow(im)
axes[1].imshow(im_filt)
format_and_render_plot()
```
- Smooth
    - reducing variability between neighboring pixals.(Blur)
    - ndi.gaussian_filter
![](https://i.imgur.com/kCTm4do.png)
```python
# Smooth "im" with Gaussian filters
im_s1 = ndi.gaussian_filter(im, sigma=1)
im_s3 = ndi.gaussian_filter(im, sigma=3)

# Draw bone masks of each image
fig, axes = plt.subplots(1,3)
axes[0].imshow(im >= 145)
axes[1].imshow(im_s1>= 145)
axes[2].imshow(im_s3>= 145)
format_and_render_plot()
```

### Detect edges
- Sobel
![](https://i.imgur.com/9UXmWEL.png)

```python
# Set weights to detect vertical edges
weights = [[1,0,-1], [1,0,-1], [1,0,-1]]

# Convolve "im" with filter weights
edges = ndi.convolve(im, weights)

# Draw the image in color
plt.imshow(edges, cmap='seismic', vmin=-150, vmax=150)
plt.colorbar()
format_and_render_plot()
```

- Sobel sum
![](https://i.imgur.com/mwe42ZD.png)

```python
# Apply Sobel filter along both axes
sobel_ax0 = ndi.sobel(im, axis=0)
sobel_ax1 = ndi.sobel(im, axis=1)

# Calculate edge magnitude 
edges = np.sqrt(np.square(sobel_ax0)+np.square(sobel_ax1))

# Plot edge magnitude
plt.imshow(edges, cmap='gray', vmax=75)
format_and_render_plot()
```

## Measurement
### Segment
```python
# Smooth intensity values
im_filt = ndi.median_filter(im,size=3)

# Select high-intensity pixels
mask_start = np.where(im_filt>60, 1, 0)
mask = ndi.binary_closing(mask_start)

# Label the objects in "mask"
labels, nlabels = ndi.label(mask)
print('Num. Labels:', nlabels)

# Create a `labels` overlay
overlay = np.where(labels > 0, labels, np.nan)

# Use imshow to plot the overlay
plt.imshow(overlay, cmap='rainbow', alpha=0.75)
format_and_render_plot()
```
![](https://i.imgur.com/S1vNOvq.png)

### Select objects

```python
# Label the image "mask"
labels, nlabels = ndi.label(mask)

# Select left ventricle pixels
lv_val = labels[128, 128]
lv_mask = np.where(labels == lv_val,1, np.nan)

# Overlay selected label
plt.imshow(lv_mask, cmap='rainbow')
plt.show()
```
![](https://i.imgur.com/uSeaRnu.png)

### Extract objects
![](https://i.imgur.com/XfRMQnh.png)

```python

# Create left ventricle mask
labels, nlabels = ndi.label(mask)
lv_val = labels[128, 128]
lv_mask = np.where(labels == lv_val, 1, 0)

# Find bounding box of left ventricle
bboxes = ndi.find_objects(lv_mask)
print('Number of objects:', len(bboxes))
print('Indices for first box:', bboxes[0])

# Crop image to the left ventricle
im_lv = im[bboxes[0]]

# Plot the cropped image
plt.imshow(im_lv)
format_and_render_plot()
```

### Measure
scipy.ndimage.measurements:

ndi.mean()
ndi.median()
ndi.sum()
ndi.maximum()
ndi.standard_deviation()
ndi.variance()
ndi.labeled_comprehension()

```python
# Variance for all pixels
var_all = ndi.variance(vol)
print('All pixels:', var_all)

# Variance for labeled pixels
var_labels = ndi.variance(vol, labels)
print('Labeled pixels:', var_labels)

# Variance for each object
var_objects = ndi.variance(vol, labels, index=[1,2])
print('Left ventricle:', var_objects[0])
print('Other tissue:', var_objects[1])
```

### Separate histograms
```python
# Create histograms for selected pixels
hist1 = ndi.histogram(vol, min=0, max=255, bins=256)
hist2 = ndi.histogram(vol, 0, 255, 256, labels)
hist3 = ndi.histogram(vol, 0, 255, 256, labels, index=1)

# Plot the histogram density
plt.plot(hist1 / hist1.sum(), label='All pixels')
plt.plot(hist2 / hist2.sum(), label='All labeled pixels')
plt.plot(hist3 / hist3.sum(), label='Left ventricle')
format_and_render_plot()
```
![](https://i.imgur.com/fkx4ixY.png)

### Calculate volume
Spatial extent is the product of:

Space occupied by each element
Number of array elements

```python
# Calculate volume per voxel
d0, d1, d2 = vol.meta['sampling']
dvoxel = d0 * d1 * d2

# Count label voxels
nvoxels=ndi.sum(1, label, index=1)

# Calculate volume of label
volume = nvoxels * dvoxel
```

### Calculate distance
Euclidean distance:
In mathematics, the Euclidean distance or Euclidean metric is the "ordinary" straight-line distance between two points in Euclidean space.
- The distance of each voxel to the nearest background value 
    - The maximum value reflect how far from the edge the most embedded point is 


```python
# Calculate left ventricle distances
lv = np.where(labels==1, 1, 0)
dists = ndi.distance_transform_edt(lv, sampling = vol.meta['sampling'])

# Report on distances
print('Max distance (mm):', ndi.maximum(dists))
print('Max location:', ndi.maximum_position(dists))

# Plot overlay of distances
overlay = np.where(dists[5] > 0, dists[5], np.nan) 
plt.imshow(overlay, cmap='hot')
format_and_render_plot()
```

### Pinpoint center of mass
- ndi.center_of_mass() returns [z, x, y] coordinates, rather than [pln, row, col] 
```python
# Extract centers of mass for objects 1 and 2
coms = ndi.center_of_mass(vol,labels,index=[1,2])
print('Label 1 center:', coms[0])
print('Label 2 center:', coms[1])

# Add marks to plot
for c0, c1, c2 in coms:
    plt.scatter(c2, c1, s=100, marker='o')
plt.show()

```

### Summarize the time series(Ejection fraction)
![](https://i.imgur.com/NZBibec.png)

Procedure

Segment left ventricle
For each 3D volume in the time series, calculate volume
Select minimum and maximum
Calculate ejection fraction

```python
# Create an empty time series
ts = np.zeros(20)

# Calculate volume at each voxel
d0, d1, d2, d3 = vol_ts.meta['sampling']
dvoxel = d1* d2*d3

# Loop over the labeled arrays
for t in range(20):
    nvoxels = ndi.sum(1, labels[t], index=1)
    ts[t] = dvoxel*nvoxels

# Plot the data
plt.plot(ts)
format_and_render_plot()
```
- Measure ejection fraction

```python
# Get index of max and min volumes
tmax = np.argmax(ts)
tmin = np.argmin(ts)

# Plot the largest and smallest volumes
fig, axes = plt.subplots(2,1)
axes[0].imshow(vol_ts[tmax, 4], vmax=160)
axes[1].imshow(vol_ts[tmin, 4], vmax=160)
format_and_render_plots()

# Calculate ejection fraction
ej_vol = max(ts)-min(ts)
ej_frac = ej_vol/max(ts)
print('Est. ejection volume (mm^3):',ej_vol)
print('Est. ejection fraction:', ej_frac)
```
