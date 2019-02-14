# [Mac OS] Parallel requires disabling default matplotlib backend
import matplotlib
matplotlib.use('agg')

import cg
from cg import glimpse
from glimpse.imports import (datetime, np, os, collections, copy)

# Required if numpy is built using OpenBLAS or MKL!
os.environ['OMP_NUM_THREADS'] = '1'

# ---- Environment ----

cg.IMAGE_PATH = '/data/brinkerhoff/glimpse_data/timelapse' # Path to time-lapse images
cg.FLAT_IMAGE_PATH = False # Whether path contains flat list of images
parallel = 32 # Number of parallel processes, True = os.cpu_count(), False = disable parallel
cylindrical = True

# ---- Tracker heuristics ----
# units: meters, days

n = 3550
# xy = (pre-computed)
xy_sigma = (2, 2)
flotation_vz_sigma = 3.5 # Multiplied by flotation probability
min_vz_sigma = 0.2 # Minimum vz_sigma, applied after flotation_vz_sigma
flotation_az_sigma = 12 # Multiplied by flotation probability
min_az_sigma = 0.2 # Minimum vz_sigma, applied after flotation_vz_sigma
tile_size = (15, 15)

# (cartesian)
# vxyz = (pre-computed from mean long-term x, y velocities and DEM slope)
# vxyz_sigma = (pre-computed from long-term x, y velocities and DEM slope)
short_vxy_sigma = 0.25 # Additional short vxy_sigma, as fraction of long vxy_sigma
axy_sigma_scale = 0.75 # Fraction of final vxy_sigma

def compute_vxyz_sigma(vxyz_sigma, flotation):
    return np.hstack((
        vxyz_sigma[0:2] * (1 + short_vxy_sigma),
        vxyz_sigma[2] + np.maximum(flotation * flotation_vz_sigma, min_vz_sigma)
    ))

def compute_axyz_sigma(vxyz_sigma, flotation):
    return np.hstack((
        vxyz_sigma[0:2] * axy_sigma_scale,
        np.maximum(flotation * flotation_az_sigma, min_az_sigma)
    ))

# (cylindrical)
# vrthz = (pre-computed from mean long-term velocities and DEM slope)
# vrthz_sigma = (pre-computed from long-term velocities and DEM slope)
short_vr_sigma = 0.25 # Additional short vr_sigma, as fraction of long vr_sigma
ar_sigma_scale = 0.75 # Fraction of final vr_sigma
dtheta_sigma = 0.05 # radians, change of theta per unit time

def compute_vrthz_sigma(vrthz_sigma, flotation):
    return np.hstack((
        vrthz_sigma[0] * (1 + short_vr_sigma),
        vrthz_sigma[1],
        vrthz_sigma[2] + np.maximum(flotation * flotation_vz_sigma, min_vz_sigma)
    ))

def compute_arthz_sigma(vrthz_sigma, flotation):
    return np.hstack((
        vrthz_sigma[0] * ar_sigma_scale,
        dtheta_sigma,
        np.maximum(flotation * flotation_az_sigma, min_az_sigma)
    ))

# ---- Load observer list ----

observer_json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)

# ---- Load DEM interpolant ----

dem_interpolant = glimpse.helpers.read_pickle('dem_interpolant.pkl')
dem_padding = 200 # m

# ---- Track points ----
start_index = 0
for i_obs in range(len(observer_json))[start_index:]:
    # ---- Load observers ----
    # observers
    observers = []
    for station, basenames in observer_json[i_obs].items():
        meta = cg.parse_image_path(basenames[0], sequence=True)
        service_calibration = cg.load_calibrations(
            station_estimate=meta['station'], station=meta['station'],
            camera=meta['camera'], merge=True, file_errors=False)
        datetimes = cg.paths_to_datetimes(basenames)
        # Use dummy Exif for speed
        service_exif = glimpse.Exif(cg.find_image(basenames[0]))
        images = []
        for basename, t in zip(basenames, datetimes):
            calibration = glimpse.helpers.merge_dicts(service_calibration,
                cg.load_calibrations(image=basename, viewdir=basename, merge=True,
                file_errors=False))
            path = cg.find_image(basename)
            image = glimpse.Image(path, cam=calibration, datetime=t, exif=service_exif)
            images.append(image)
        # NOTE: Determine sigma programmatically?
        observer = glimpse.Observer(images, cache=True, correction=True, sigma=0.3)
        observers.append(observer)
    # ---- Load track points ----
    t = min([observer.datetimes[0] for observer in observers])
    datestr = t.strftime('%Y%m%d')
    basename = datestr + '-' + str(i_obs)
    if cylindrical:
        # ids, xy, observer_mask, vxyz, vxyz_sigma, flotation
        params = glimpse.helpers.read_pickle(
            os.path.join('points-cylindrical', basename + '.pkl'))
    else:
        # ids, xy, observer_mask, vrthz, vrthz_sigma, flotation
        params = glimpse.helpers.read_pickle(
            os.path.join('points-cartesian', basename + '.pkl'))
    # ---- Load DEM ----
    # dem, dem_sigma
    dem, dem_sigma = dem_interpolant(t, return_sigma=True)
    # Crop DEM (for lower memory use)
    box = (glimpse.helpers.bounding_box(params['xy']) +
        np.array([-1, -1, 1, 1]) * dem_padding)
    dem.crop(xlim=box[0::2], ylim=box[1::2])
    dem_sigma.crop(xlim=box[0::2], ylim=box[1::2])
    dem.crop_to_data()
    dem_sigma.crop_to_data()
    # ---- Compute motion models ----
    # motion_models
    time_unit = datetime.timedelta(days=1)
    m = len(params['xy'])
    if cylindrical:
        vrthz_sigmas = [compute_vrthz_sigma(
            params['vrthz_sigma'][i], params['flotation'][i]) for i in range(m)]
        arthz_sigmas = [compute_arthz_sigma(
            vrthz_sigmas[i], params['flotation'][i]) for i in range(m)]
        motion_models = [glimpse.tracker.CylindricalMotionModel(
            n=n, dem=dem, dem_sigma=dem_sigma, time_unit=time_unit,
            xy_sigma=xy_sigma, xy=params['xy'][i], vrthz=params['vrthz'][i],
            vrthz_sigma=vrthz_sigmas[i], arthz_sigma=arthz_sigmas[i])
            for i in range(m)]
    else:
        vxyz_sigmas = [compute_vxyz_sigma(
            params['vxyz_sigma'][i], params['flotation'][i]) for i in range(m)]
        axyz_sigmas = [compute_axyz_sigma(
            vxyz_sigmas[i], params['flotation'][i]) for i in range(m)]
        motion_models = [glimpse.tracker.CartesianMotionModel(
            n=n, dem=dem, dem_sigma=dem_sigma, time_unit=time_unit,
            xy_sigma=xy_sigma, xy=params['xy'][i], vxyz=params['vxyz'][i],
            vxyz_sigma=vxyz_sigmas[i], axyz_sigma=axyz_sigmas[i])
            for i in range(m)]
    # ---- Track points ----
    # tracks, tracks_r
    tracker = glimpse.Tracker(observers=observers)
    # Initialize placeholders
    # forward | forward + last vxy | reverse | reverse + last vxy
    suffixes = ('f', 'fv', 'r', 'rv')
    directions = (1, 1, -1, -1)
    tracks = [None] * len(suffixes)
    if cylindrical:
        paths = [os.path.join(
            'tracks-cylindrical', basename + '-' + suffix + '.pkl')
            for suffix in suffixes]
    else:
        paths = [os.path.join(
            'tracks-cartesian', basename + '-' + suffix + '.pkl')
            for suffix in suffixes]
    is_file = [os.path.isfile(path) for path in paths]
    # Run forward and backward
    for i in (0, 2):
        if not is_file[i]:
            print(basename + '-' + suffixes[i])
            tracks[i] = tracker.track(motion_models=motion_models,
                observer_mask=params['observer_mask'], tile_size=tile_size,
                parallel=parallel, datetimes=tracker.datetimes[::directions[i]])
        if not is_file[i + 1]:
            print(basename + '-' + suffixes[i + 1])
            if not tracks[i]:
                tracks[i] = glimpse.helpers.read_pickle(paths[i])
            # Start with last vx, vy distribution of first run
            mask, first, last = tracks[i].endpoints()
            last_vxy = tracks[i].vxyz[mask, last, 0:2]
            last_vxy_sigma = tracks[i].vxyz_sigma[mask, last, 0:2]
            vxy_motion_models = [copy.copy(model) for model in motion_models]
            if cylindrical:
                for j, model in enumerate(np.array(vxy_motion_models)[mask]):
                    # Estimate cylindrical priors from cartesian results
                    vxy = last_vxy[j] + last_vxy_sigma[j] * np.random.randn(100, 2)
                    speed = np.hypot(vxy[:, 0], vxy[:, 1])
                    vr, vr_sigma = speed.mean(), speed.std()
                    thetas = np.arctan2(vxy[:, 1], vxy[:, 0])
                    unit_yx = (
                        np.sin(thetas).mean(),
                        np.cos(thetas).mean())
                    theta = np.arctan2(*unit_yx)
                    theta_sigma = np.sqrt(-2 * np.log(np.hypot(*unit_yx)))
                    model.vrthz = np.hstack((vr, theta, model.vrthz[2]))
                    model.vrthz_sigma = np.hstack((vr_sigma, theta_sigma,
                        model.vrthz_sigma[2]))
            else:
                for j, model in enumerate(np.array(vxy_motion_models)[mask]):
                    model.vxyz = np.hstack((last_vxy[j], model.vxyz[2]))
                    model.vxyz_sigma = np.hstack((last_vxy_sigma[j], model.vxyz_sigma[2]))
            # Repeat track
            tracks[i + 1] = tracker.track(motion_models=vxy_motion_models,
                observer_mask=params['observer_mask'], tile_size=tile_size,
                parallel=parallel, datetimes=tracker.datetimes[::directions[i + 1]])
    # ---- Clean up tracks ----
    # Clean up tracker, since saved in Tracks.tracker
    tracker.reset()
    # Clear cached images, since saved in Tracks.tracker.observers
    for observer in observers:
        observer.clear_images()
    # Clear motion_models from Tracks.params
    for track in tracks:
        track.params['motion_models'] = None
    # ---- Save tracks to file ----
    for i in range(len(suffixes)):
        if not is_file[i] and tracks[i] is not None:
            glimpse.helpers.write_pickle(tracks[i], paths[i])
