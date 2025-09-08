import numpy as np
import tifffile
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

# Public MICrONS segmentation
SOURCE = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"

# Make a CloudVolume (mip 0 by default)
vol = CloudVolume(SOURCE, progress=True, cache=False, fill_missing=True)

# Define a SMALL region first (x, y, z) in voxel coords
start_xyz = (161198/2, 181692/2, 9752/2)
end_xyz   = (227205/2, 239179/2, 23137/2)  # end is exclusive

bbox = Bbox(start_xyz, end_xyz)

# CloudVolume slicing uses (z, y, x)
cutout = vol[bbox.to_slices()]   # shape typically (z, y, x, 1) or (z, y, x)
cutout = np.squeeze(cutout)      # drop channel axis if present

# Save to TIFF (BigTIFF if large); segmentation labels are usually uint64
tifffile.imwrite("segmentation_mask.tif", cutout, bigtiff=True)
print("Saved segmentation_mask.tif with shape", cutout.shape, cutout.dtype)
