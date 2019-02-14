import cg
from cg import glimpse
import glimpse.unumpy as unp
import scipy.stats
from glimpse.imports import (datetime, np, os, shapely, re, matplotlib, collections)
import glob
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')

grid_step = (100, 100) # m
max_distance = 10e3 # m
bed_sigma = 20 # m
density_water = 1025 # kg / m^3
density_ice = 916.7 # kg / m^3

# ---- Load first image from each observer station ----
# images

json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)
start_images = []
progress = glimpse.helpers._progress_bar(max=len(json))
for observers in json:
    starts = []
    for station, basenames in observers.items():
        ids = cg.parse_image_path(basenames[0], sequence=True)
        cam_args = cg.load_calibrations(station=station, camera=ids['camera'],
            image=basenames[0], viewdir=basenames[0], merge=True,
            file_errors=False)
        path = cg.find_image(basenames[0])
        starts.append(glimpse.Image(path, cam=cam_args))
    start_images.append(tuple(starts))
    progress.next()

# ---- Load DEM interpolant ----

dem_interpolant = glimpse.helpers.read_pickle('dem_interpolant.pkl')

# ---- Load canonical velocities (cartesian) ----
# vx, vx_sigma, vy, vy_sigma

names = 'vx', 'vx_stderr', 'vy', 'vy_stderr'
vx, vx_sigma, vy, vy_sigma = [glimpse.Raster.read(
    os.path.join('velocity', name + '.tif'))
    for name in names]

# ---- Load canonical velocities (cylindrical) ----
# vr, vr_sigma, theta, theta_sigma

names = 'vr', 'vr_stderr', 'theta', 'theta_stderr'
vr, vr_sigma, theta, theta_sigma = [glimpse.Raster.read(
    os.path.join('velocity', name + '.tif'))
    for name in names]

# ---- Load canonical bed ----
# bed

bed = glimpse.Raster.read('bed.tif')

# ---- Build track template ----

grid = glimpse.helpers.box_to_grid(dem_interpolant.means[0].box2d,
    step=grid_step, snap=(0, 0))
track_template = glimpse.Raster(np.ones(grid[0].shape, dtype=bool),
    x=grid[0], y=grid[1][::-1])
xy = glimpse.helpers.grid_to_points((track_template.X, track_template.Y))
selected = glimpse.helpers.points_in_polygon(xy, cg.Glacier())
# Filter by velocity availability
# NOTE: Use nearest to avoid NaN propagation (and on same grid anyway)
selected &= ~np.isnan(vx.sample(xy, order=0))
mask = selected.reshape(track_template.shape)
track_points = xy[mask.ravel()]
track_ids = np.ravel_multi_index(np.nonzero(mask), track_template.shape)

# Write to file
track_template.Z &= mask
track_template.Z = track_template.Z.astype(np.uint8)
track_template.write(os.path.join('points-cartesian', 'template.tif'), crs=32606)
track_template.write(os.path.join('points-cylindrical', 'template.tif'), crs=32606)

# ---- For each observer ----

for obs in range(len(start_images)):
    print(obs)
    images = start_images[obs]
    # Check within DEM interpolant bounds
    t = np.min([img.datetime for img in images])
    if t > np.max(dem_interpolant.x):
        raise ValueError('Images begin after last DEM')
    # -- Load DEM --
    dem, dem_sigma = dem_interpolant(t, return_sigma=True)
    # Compute union of camera view boxes
    boxes = []
    for img in images:
        dxyz = img.cam.invproject(img.cam.edges(step=10))
        scale = max_distance / np.linalg.norm(dxyz[:, 0:2], axis=1)
        xyz = np.vstack((img.cam.xyz, img.cam.xyz + dxyz * scale.reshape(-1, 1)))
        boxes.append(glimpse.helpers.bounding_box(xyz[:, 0:2]))
    box = glimpse.helpers.union_boxes(boxes)
    # Mask camera foreground for viewshed calculation
    for img in images:
        dem.fill_circle(center=img.cam.xyz, radius=400, value=np.nan)
    # --- Load track points and observer mask ---
    # Load glacier polygon
    ij = dem_interpolant.nearest(t)
    polygons = [glimpse.helpers.box_to_polygon(box)]
    polygons += [dem_interpolant.means[i].polygon for i in ij]
    polygons += [cg.load_glacier_polygon(t)]
    polygon = cg.intersect_polygons(polygons)
    observer_mask = cg.select_track_points(track_points, images=images,
        polygon=polygon, dem=dem, max_distance=max_distance)
    selected = np.count_nonzero(observer_mask, axis=1) > 0
    # xy (n, ), observer_mask (n, o)
    xy, obsmask, ids = track_points[selected], observer_mask[selected], track_ids[selected]
    # ---- Compute motion parameters (cartesian) ----
    n = len(xy)
    # vxyz | vxyz_sigma
    vxyz = np.ones((n, 3), dtype=float)
    vxyz_sigma = np.ones((n, 3), dtype=float)
    # (x, y): Sample from velocity grids
    vxyz[:, 0] = vx.sample(xy, order=0)
    vxyz[:, 1] = vy.sample(xy, order=0)
    vxyz_sigma[:, 0] = vx_sigma.sample(xy, order=0)
    vxyz_sigma[:, 1] = vy_sigma.sample(xy, order=0)
    # (z): Compute by integrating dz/dx and dz/dy over vx and vy
    rowcol = dem.xy_to_rowcol(xy, snap=True)
    dz = np.dstack(dem.gradient())[rowcol[:, 0], rowcol[:, 1], :]
    # sigma for dz/dx * vx + dz/dy * vy, assume zi, zj are fully correlated
    udz = unp.uarray(dz, sigma=None)
    uvxy = unp.uarray(vxyz[:, 0:2], vxyz_sigma[:, 0:2])
    vxyz[:, 2], vxyz_sigma[:, 2] = (
        udz[:, 0] * uvxy[:, 0] + udz[:, 1] * uvxy[:, 1]).tuple()
    # ---- Compute motion parameters (cylindrical) ----
    n = len(xy)
    # vrthz | vrthz_sigma
    vrthz = np.ones((n, 3), dtype=float)
    vrthz_sigma = np.ones((n, 3), dtype=float)
    # (r, theta): Sample from velocity grids
    vrthz[:, 0] = vr.sample(xy, order=0)
    vrthz[:, 1] = theta.sample(xy, order=0)
    vrthz_sigma[:, 0] = vr_sigma.sample(xy, order=0)
    vrthz_sigma[:, 1] = theta_sigma.sample(xy, order=0)
    # (z): Reuse cartesian results
    vrthz[:, 2], vrthz_sigma[:, 2] = vxyz[:, 2], vxyz_sigma[:, 2]
    # ---- Determine probability of flotation ----
    # vz_sigma, az_sigma: Typically very small, but large if glacier floating
    zw = unp.uarray(16, 2) # m, mean HAE
    Zs = unp.uarray(dem.sample(xy, order=1), dem_sigma.sample(xy, order=1)) # m
    Zb = unp.uarray(bed.sample(xy, order=1), bed_sigma) # m
    hmax = Zs - Zb
    hf = (zw - Zb) * (density_water / density_ice)
    # probability hf > hmax
    # https://math.stackexchange.com/a/40236
    dh = hmax - hf
    flotation = scipy.stats.norm().cdf(-dh.mean / dh.sigma)
    # ---- Save parameters to file ----
    basename = t.strftime('%Y%m%d') + '-' + str(obs) + '.pkl'
    # ids, xy, observer_mask
    # flotation: Probability of flotation
    # vxyz, vxyz_sigma (cartesian)
    glimpse.helpers.write_pickle(
        dict(ids=ids, xy=xy, observer_mask=obsmask, flotation=flotation,
        vxyz=vxyz, vxyz_sigma=vxyz_sigma),
        path=os.path.join('points-cartesian', basename))
    # vrthz, vrthz_sigma (cylindrical)
    glimpse.helpers.write_pickle(
        dict(ids=ids, xy=xy, observer_mask=obsmask, flotation=flotation,
        vrthz=vrthz, vrthz_sigma=vrthz_sigma),
        path=os.path.join('points-cylindrical', basename))

# ---- Plotting ----

# # Plot (motion)
# matplotlib.pyplot.figure()
# dem.plot(dem.hillshade(), cmap='gray')
# dem.set_plot_limits()
# matplotlib.pyplot.plot(polygon[:, 0], polygon[:, 1], color='gray')
# matplotlib.pyplot.quiver(xy[:, 0], xy[:, 1], vxyz[:, 0], vxyz[:, 1],
#     angles='xy', color='red')
# # matplotlib.pyplot.quiver(xy[:, 0], xy[:, 1], vxyz_sigma[:, 0], vxyz_sigma[:, 1],
# #     angles='xy', color='red')
# # raster template
# temp = track_template.copy()
# temp.Z = np.full(temp.shape, np.nan)
# # plot data as raster
# r = temp.copy()
# r.Z.flat[ids] = flotation
# # r.Z.flat[ids] = np.hypot(vxyz[:, 0], vxyz[:, 1])
# # r.Z.flat[ids] = np.hypot(vxyz_sigma[:, 0], vxyz_sigma[:, 1])
# r.plot()
# # r.plot(vmin=0, vmax=2, cmap='bwr')
# matplotlib.pyplot.colorbar()

# # Plot (map)
# matplotlib.pyplot.figure()
# dem.plot(dem.hillshade(), cmap='gray')
# for i, obs in enumerate(observers):
#     viewpoly = obs.images[0].cam.viewpoly(max_depth, step=100)[:, 0:2]
#     matplotlib.pyplot.plot(viewpoly[:, 0], viewpoly[:, 1], color='black')
#     matplotlib.pyplot.plot(xy[mask[:, i], 0], xy[mask[:, i], 1],
#         marker='.', linestyle='none')
# matplotlib.pyplot.savefig('temp.png', dpi=100)

# # Plot (image)
# for i, obs in enumerate(observers):
#     matplotlib.pyplot.figure()
#     xyz = np.column_stack((xy[mask[:, i]], dem.sample(xy[mask[:, i]])))
#     uv = obs.images[0].cam.project(xyz, correction=True)
#     obs.images[0].plot()
#     matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], color='red', marker='.',
#         linestyle='none')
#     matplotlib.pyplot.savefig('temp-' + str(i) + '.png', dpi=100)
