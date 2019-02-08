import cg
from cg import glimpse
from glimpse.imports import (np, os, datetime, re)
import glob
from merge_tracks_functions import *

# ---- Constants ----

# Input
tracks_path = 'tracks'
points_path = 'points'
labels = 'f', 'fv', 'r', 'rv'

# Output
arrays_path = 'tracks-arrays'
rasters_path = 'tracks-rasters'

# Knobs & Switches
flatten_function = flatten_tracks_ethan # Returns vxyz means, sigmas with dimensions [point] . [x | y | z]
mean_datetimes = True # True: mean of times, False: middle of time range
# Spatial median filter
diagonal_neighbors = False # Whether to include diagonal neighbors
min_observers = 2 # Median ignores neighbors with observers below min if any above min
fill_missing = False # Whether to fill missing points with neighborhood median
nan_median = True # Whether to ignore missing values in computing median
exact_median_variance = False # Whether to use "exact" variance of median calculation (when averaging nearest neighbors)

# ---- Load medatada ----
# basenames: <date>.<interval id>
# template:
# ids: Unique point ids (flat index of points in template raster)

# Load tracks basenames
# ['20040624-0', '20040625-1', ...]
paths = glob.glob(os.path.join(tracks_path, '*.pkl'))
basepaths = [glimpse.helpers.strip_path(path) for path in paths]
basenames = np.unique([re.sub("-[^-]+$", '', path) for path in basepaths
    if re.search('^[0-9]{8}-[0-9]+-.+', path) is not None])

# Load template
template = glimpse.Raster.read(os.path.join(points_path, 'template.tif'))

# Load ids of all tracked points
ids = []
for basename in basenames:
    points = glimpse.helpers.read_pickle(
        os.path.join(points_path, basename + '.pkl'))
    ids = np.union1d(ids, points['ids']).astype(int)

# ---- Build arrays ----
# xyi: [ x | y | id ]
# datetimes: [ t ]
# means: [ point ] . [ t ] . [ vx | vy | vz ]
# sigmas: [ point ] . [ t ] . [ vx_sigma | vy_sigma | vz_sigma ]
# nobservers: [ point ] . [ t ]
# flotation: [ point ] . [ t ]

# Initialize arrays
idx = range(len(basenames))
datetimes = []
shape = len(ids), len(idx), 3
means = np.full(shape, np.nan, dtype=np.float32)
sigmas = means.copy()
flotation = means[..., 0].copy()
nobservers = np.zeros(shape[0:2], dtype=np.uint8)

# Load flattened tracks into arrays
origin = datetime.datetime(1970, 1, 1, 0, 0, 0)
for col, basename in enumerate(basenames):
    print(basename)
    points = glimpse.helpers.read_pickle(
        os.path.join(points_path, basename + '.pkl'))
    runs = {label: glimpse.helpers.read_pickle(
        os.path.join(tracks_path, basename + '-' + label + '.pkl'))
        for label in labels}
    rows = np.searchsorted(ids, points['ids'])
    means[rows, col], sigmas[rows, col] = flatten_function(runs)
    flotation[rows, col] = points['flotation']
    nobservers[rows, col] = points['observer_mask'].sum(axis=1)
    if mean_datetimes:
        # Midtime as mean
        datetimes.append(origin + (runs['f'].datetimes - origin).mean())
    else:
        # Midtime as middle
        datetimes.append(origin + (runs['f'].datetimes[[1, -1]] - origin).mean())

# Precompute spatial neighborhoods and masks
ncols = template.shape[1]
neighbor_ids = np.column_stack((ids, ids - 1, ids + 1, ids - ncols, ids + ncols))
if diagonal_neighbors:
    neighbor_ids = np.column_stack((neighbor_ids, np.column_stack((
        ids - 1 - cols, ids + 1 - cols, ids - 1 + cols, ids + 1 + cols))))
neighbor_rows = np.searchsorted(ids, neighbor_ids)
missing = ~np.isin(neighbor_ids, ids)
neighbor_rows[missing] = 0
few_cams = ((nobservers[neighbor_rows, :] < min_observers) &
    (nobservers[neighbor_rows, :].max(axis=1, keepdims=True) >= min_observers))
isnan = np.isnan(means)

# Apply spatial median filter
fmeans = np.full(means.shape, np.nan, dtype=np.float32)
fsigmas = fmeans.copy()
for dim in range(means.shape[2]):
    print(dim)
    # (point, neighbor, time)
    m = means[..., dim][neighbor_rows, :]
    s = sigmas[..., dim][neighbor_rows, :]
    # Mask out missing neighbors or neighbors with too few cameras
    m[missing] = np.nan
    m[few_cams] = np.nan
    if not fill_missing:
        # Mask out neighborhoods with missing centers
        m[np.tile(isnan[..., 0][:, None, :], (1, m.shape[1], 1))] = np.nan
    if not nan_median:
        # Mask out neighborhoods with missing neighbors
        m[np.tile(np.any(np.isnan(m), axis=1, keepdims=True), (1, m.shape[1], 1))] = np.nan
    # Compute median of means
    medians = np.nanmedian(m, axis=1, keepdims=True)
    # Compute sigma of median of means
    median_diff = np.abs((m - medians))
    is_median = median_diff == np.nanmin(median_diff, axis=1, keepdims=True)
    m[~is_median] = np.nan
    s[~is_median] = np.nan
    # Single median: Select mean, sigma of median neighbor
    is_single_median = is_median & (np.sum(is_median, axis=1, keepdims=True) == 1)
    mask = is_single_median.any(axis=1)
    fmeans[mask, dim] = m[is_single_median]
    fsigmas[mask, dim] = s[is_single_median]
    # Multiple median: Take unweighted average (correlation = 1)
    is_multiple_median = is_median & ~is_single_median
    mask = is_multiple_median.any(axis=1)
    fmeans[mask, dim] = medians.squeeze(axis=1)[mask]
    if exact_median_variance:
        # "Exact" variance
        n = s.shape[1]
        pairs = np.triu_indices(n=n, k=1)
        sqweights = 1 / np.sum(is_multiple_median, axis=1, keepdims=False)**2
        variances = sqweights * np.nansum(s**2, axis=1, keepdims=False)
        variances += 2 * sqweights * np.nansum(
            np.take(s, pairs[0], axis=1) *
            np.take(s, pairs[1], axis=1), axis=1, keepdims=False)
        fsigmas[mask, dim] = np.sqrt(variances[mask])
    else:
        # Approximate variance using the mean of the variances
        fsigmas[mask, dim] = np.sqrt(np.nanmean(s**2, axis=1)[mask])

# Write files
datetimes = np.asarray(datetimes)
glimpse.helpers.write_pickle(
    datetimes, os.path.join(arrays_path, 'datetimes.pkl'))
glimpse.helpers.write_pickle(fmeans, os.path.join(arrays_path, 'means.pkl'))
glimpse.helpers.write_pickle(fsigmas, os.path.join(arrays_path, 'sigmas.pkl'))
nobservers[np.isnan(means[..., 0])] = 0
glimpse.helpers.write_pickle(
    nobservers, os.path.join(arrays_path, 'nobservers.pkl'))
glimpse.helpers.write_pickle(
    flotation, os.path.join(arrays_path, 'flotation.pkl'))
# (xyi)
xy = glimpse.helpers.grid_to_points((template.X, template.Y))[ids]
xyi = np.column_stack((xy, ids))
glimpse.helpers.write_pickle(xyi, os.path.join(arrays_path, 'xyi.pkl'))
np.savetxt(
    fname=os.path.join(arrays_path, 'xyi.csv'), X=xyi,
    delimiter=',', fmt='%d', header='x,y,id', comments='')

# ---- Build rasters ----
# [ y ] . [ x ] . [ t ]

# Prepare base raster
raster = template.copy()
raster.Z = raster.Z.astype(np.float32)
raster.Z[:] = np.nan
raster.Z.flat[xyi[:, 2].astype(int)] = True
raster.crop_to_data()
rowcols = raster.xy_to_rowcol(xyi[:, 0:2], snap=True)

# Rasterize arrays
base = np.full(raster.shape + (len(datetimes), ), np.nan, dtype=np.float32)
for i, basename in enumerate(('vx', 'vy', 'vz')):
    base[rowcols[:, 0], rowcols[:, 1]] = means[..., i]
    glimpse.helpers.write_pickle(base, os.path.join(rasters_path, basename + '.pkl'))
for i, basename in enumerate(('vx_sigma', 'vy_sigma', 'vz_sigma')):
    base[rowcols[:, 0], rowcols[:, 1]] = sigmas[..., i]
    glimpse.helpers.write_pickle(base, os.path.join(rasters_path, basename + '.pkl'))
base[rowcols[:, 0], rowcols[:, 1]] = flotation
glimpse.helpers.write_pickle(base, os.path.join(rasters_path, 'flotation.pkl'))
base[rowcols[:, 0], rowcols[:, 1]] = nobservers
base[np.isnan(base)] = 0
glimpse.helpers.write_pickle(base.astype(np.uint8), os.path.join(rasters_path, 'nobservers.pkl'))

# Write template
raster.Z[np.isnan(raster.Z)] = 0
raster.Z = raster.Z.astype(np.uint8)
raster.write(os.path.join(rasters_path, 'template.tif'), crs=32606)

# Write metadata
glimpse.helpers.write_pickle(datetimes, os.path.join(rasters_path, 'datetimes.pkl'))
glimpse.helpers.write_pickle(xyi, os.path.join(rasters_path, 'xyi.pkl'))

# ---- Build strain rates ----

vx = glimpse.helpers.read_pickle(os.path.join(rasters_path, 'vx.pkl'))
vy = glimpse.helpers.read_pickle(os.path.join(rasters_path, 'vy.pkl'))
xout, yout, xin, yin = compute_principal_strains(vx, vy, template.d)

# Save as rasters
glimpse.helpers.write_pickle(xout, os.path.join(rasters_path, 'extension_x.pkl'))
glimpse.helpers.write_pickle(yout, os.path.join(rasters_path, 'extension_y.pkl'))
glimpse.helpers.write_pickle(xin, os.path.join(rasters_path, 'compression_x.pkl'))
glimpse.helpers.write_pickle(yin, os.path.join(rasters_path, 'compression_y.pkl'))

# Save as arrays
template = glimpse.Raster.read(os.path.join(rasters_path, 'template.tif'))
rows, cols = np.nonzero(template.Z)
glimpse.helpers.write_pickle(
    np.dstack((xout[rows, cols, :], yout[rows, cols, :])),
    os.path.join(arrays_path, 'extension.pkl'))
glimpse.helpers.write_pickle(
    np.dstack((xin[rows, cols, :], yin[rows, cols, :])),
    os.path.join(arrays_path, 'compression.pkl'))
