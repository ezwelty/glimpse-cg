import cg
from cg import glimpse
from glimpse.imports import (datetime, np, os, collections)
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')

surface_sigma = 3 # m, surface roughness
grid_size = 20 # m
zlim = (1, np.inf) # m
fill_crevasses_args = dict(
    maximum=dict(size=5), # 100 m
    gaussian=dict(sigma=5), # 200 m (68%)
    mask=lambda x: ~np.isnan(x),
    fill=True)
max_distance = 10e3 # m
dem_interpolant_path = 'dem_interpolant.pkl'

# ---- Build DEM template ----

json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)
stations = set([station for x in json for station in x])
station_xy = np.vstack([f['geometry']['coordinates'][:, 0:2]
    for station, f in cg.Stations().items()
    if station in stations])
box = glimpse.helpers.bounding_box(cg.Glacier())
XY = glimpse.helpers.box_to_grid(box, step=(grid_size, grid_size),
    snap=(0, 0), mode='grid')
xy = glimpse.helpers.grid_to_points(XY)
distances = glimpse.helpers.pairwise_distance(xy, station_xy, metric='euclidean')
selected = distances.min(axis=1) < max_distance
box = glimpse.helpers.bounding_box(xy[selected]) + 0.5 * np.array([-1, -1, 1, 1]) * grid_size
shape = np.diff([box[1::2], box[0::2]], axis=1) / grid_size
dem_template = glimpse.Raster(np.ones(shape.astype(int).ravel(), dtype=bool),
    x=box[0::2], y=box[1::2][::-1])
dem_points = glimpse.helpers.grid_to_points((dem_template.X, dem_template.Y))

# ---- Select DEMs ----

dem_sigmas = {
    'aerometric': 1.5,
    'ifsar': 1.5 + 0.5, # additional time uncertainty
    'arcticdem': 3,
    'tandem': 3 # after bulk corrections
}
dem_keys = [
    ('20040618', 'aerometric'),
    ('20040707', 'aerometric'),
    ('20050811', 'aerometric'),
    ('20050827', 'aerometric'),
    ('20060712', 'aerometric'),
    ('20060727', 'aerometric'),
    ('20070922', 'aerometric'),
    ('20080811', 'aerometric'),
    ('20090803', 'aerometric'),
    ('20090827', 'aerometric'),
    ('20100525', 'aerometric'),
    ('20100602', 'aerometric'),
    ('20100720', 'ifsar'), # +- 10 days mosaic
    ('20100906', 'arcticdem'),
    ('20110618', 'tandem'),
    ('20110721', 'tandem'),
    ('20110812', 'tandem'),
    ('20110903', 'tandem'),
    ('20111211', 'tandem'),
    ('20120102', 'tandem'),
    ('20120204', 'tandem'),
    ('20120308', 'tandem'),
    ('20120329', 'arcticdem'),
    ('20120507', 'arcticdem'),
    ('20120617', 'arcticdem'),
    ('20120717', 'arcticdem'),
    ('20120813', 'arcticdem'),
    ('20121012', 'arcticdem'),
    ('20121123', 'arcticdem'),
    ('20130326', 'arcticdem'),
    ('20130610', 'arcticdem'),
    ('20130712', 'arcticdem'),
    ('20131119', 'arcticdem'),
    ('20140417', 'tandem'),
    # ('20140419', 'arcticdem'), # Coverage too small
    ('20140531', 'tandem'),
    ('20140622', 'tandem'),
    ('20140703', 'tandem'),
    ('20150118', 'arcticdem'),
    ('20150227', 'arcticdem'),
    ('20150423', 'arcticdem'),
    ('20150527', 'arcticdem'),
    ('20150801', 'arcticdem'),
    ('20150824', 'arcticdem'),
    ('20150930', 'arcticdem'),
    ('20160614', 'arcticdem'),
    ('20160820', 'arcticdem')]
dem_keys.sort(key=lambda x: x[0])

# ---- Build DEM interpolant ----

# Compute means and sigmas
means, sigmas = [], []
for datestr, demtype in dem_keys:
    print(datestr, demtype)
    path = os.path.join(root, 'dem-' + demtype, 'data', datestr + '.tif')
    # HACK: Aerial and satellite imagery taken around local noon (~ 22:00 UTC)
    t = datetime.datetime.strptime(datestr + str(22), '%Y%m%d%H')
    dem = glimpse.Raster.read(path,
        xlim=dem_template.xlim + np.array((-1, 1)) * grid_size,
        ylim=dem_template.ylim + np.array((1, -1)) * grid_size,
        d=grid_size)
    dem.crop(zlim=zlim)
    z = dem.sample(dem_points, order=1, bounds_error=False).reshape(dem_template.shape)
    dem = glimpse.Raster(z, x=dem_template.xlim, y=dem_template.ylim, datetime=t)
    # Cache dem type and glacier polygon
    dem.type = demtype
    dem.polygon = cg.load_glacier_polygon(t=dem.datetime, demtype=dem.type)
    # Mask forebay
    forebay = cg.load_forebay_polygon(glacier=dem.polygon)
    mask = dem.rasterize_poygons([forebay])
    dem.Z[mask] = np.nan
    dem.fill_crevasses(**fill_crevasses_args)
    dem.Z[mask] = np.nan
    # Add to results
    means.append(dem)
    sigma = np.sqrt(dem_sigmas[demtype]**2 + surface_sigma**2)
    sigmas.append(sigma)

# Initialize interpolant
dem_interpolant = glimpse.RasterInterpolant(means=means, sigmas=sigmas,
    x=[dem.datetime for dem in means])

# Write to file
glimpse.helpers.write_pickle(dem_interpolant, dem_interpolant_path)
