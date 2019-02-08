import cg
from cg import glimpse
from glimpse.imports import (np, os, datetime, re)
import matplotlib.pyplot
import mpl_toolkits.axes_grid1

# ---- Constants ----

rasters_path = 'tracks-rasters'
min_observers = 1

# ---- Load data -----

# Read rasters
datetimes = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'datetimes.pkl'))
vx = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'vx.pkl'))
vy = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'vy.pkl'))
vz = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'vz.pkl'))
nobservers = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'nobservers.pkl'))
flotation = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'flotation.pkl'))
template = glimpse.Raster.read(os.path.join(rasters_path, 'template.tif'))
extension_x = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'extension_x.pkl'))
extension_y = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'extension_y.pkl'))
compression_x = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'compression_x.pkl'))
compression_y = glimpse.helpers.read_pickle(
    os.path.join(rasters_path, 'compression_y.pkl'))

# Crop to coverage
raster = template.copy()
raster.Z = np.where((nobservers >= min_observers).any(axis=2), True, np.nan)
point_mask = raster.data_extent()
time_mask = (nobservers >= min_observers).any(axis=(0, 1))
indices = point_mask + (time_mask, )
raster.crop_to_data()

# Mask data
# dy, dx = np.gradient(nobservers, axis=(0, 1))
# few_obs = (nobservers < min_observers) | (dx != 0) | (dy != 0)
few_obs = (nobservers < min_observers)
# nobservers = nobservers.astype(float)
# nobservers[nobservers == 0] = np.nan
# few_obs |= (scipy.ndimage.minimum_filter(nobservers, size=(3, 3, 1)) < min_observers)
vx[few_obs] = np.nan
vy[few_obs] = np.nan
vz[few_obs] = np.nan
extension_x[few_obs] = np.nan
extension_y[few_obs] = np.nan
compression_x[few_obs] = np.nan
compression_y[few_obs] = np.nan
flotation[few_obs] = np.nan

# Compute speeds
speeds = np.sqrt(vx**2 + vy**2)

# ---- Animate speeds ----

i = 0
fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(12, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
ax.set_aspect(1)
im = ax.imshow(speeds[indices][..., i], vmin=0, vmax=20,
    extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", "5%", pad="3%")
matplotlib.pyplot.colorbar(im, cax=cax)
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)

def update_plot(i):
    print(i)
    im.set_array(speeds[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return im, txt

ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('speed_multi2.mp4')

# ---- Animate velocity vectors ----

# Plot vx, vy
fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(12, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
ax.set_aspect(1)
i = 0
scale = 15
mask = ~np.isnan(vx[indices][..., i])
quiver = matplotlib.pyplot.quiver(
    raster.X, raster.Y, vx[indices][..., i] * scale, vy[indices][..., i] * scale,
    # color='black',
    speeds[indices][..., i], clim=[0, 20],
    alpha=1, width=5, headaxislength=0, headwidth=1, minlength=0,
    pivot='tail', angles='xy', scale_units='xy', scale=1)
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", "5%", pad="3%")
matplotlib.pyplot.colorbar(quiver, cax=cax)
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)

def update_plot(i):
    print(i)
    quiver.set_UVC(vx[indices][..., i] * scale, vy[indices][..., i] * scale, speeds[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return quiver, txt

ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('vxy.mp4', dpi=150)

# ---- Animate principal strains ----

fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(8, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
ax.set_aspect(1)
i = 0
scale = 15
strain_scale = 1e4
# quiver_velocity = ax.quiver(
#     raster.X, raster.Y, vx[indices][..., i] * scale, vy[indices][..., i] * scale,
#     color='black', alpha=0.25, width=5, headaxislength=0, headwidth=1,
#     minlength=0, pivot='tail', angles='xy', scale_units='xy', scale=1,
#     zorder=0)
# im_vz = ax.imshow(vz[indices][..., i] - vz_slope[indices][..., i],
#     cmap=cmocean.cm.balance, vmin=-1, vmax=1, zorder=0,
#     extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
im_vz = ax.imshow(vz[indices][..., i],
    cmap=cmocean.cm.balance, vmin=-2, vmax=2, zorder=0,
    extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
# matplotlib.pyplot.colorbar(im_vz)
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
kwargs = dict(
    angles='xy', scale=1, scale_units='xy', units='xy', width=20,
    headwidth=0, headlength=0, headaxislength=0)
quiver_extension = ax.quiver(
    raster.X, raster.Y,
    strain_scale * extension_x[indices][..., i], strain_scale * extension_y[indices][..., i],
    color='black', pivot='mid', zorder=2, **kwargs)
quiver_compression = ax.quiver(
    raster.X, raster.Y,
    strain_scale * compression_x[indices][..., i], strain_scale * compression_y[indices][..., i],
    color='red', pivot='mid', zorder=1, **kwargs)
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)
# matplotlib.pyplot.legend(('velocity', 'extension', 'compression'))
matplotlib.pyplot.legend(('extension', 'compression'))
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])

def update_plot(i):
    print(i)
    # quiver_velocity.set_UVC(vx[indices][..., i] * scale, vy[indices][..., i] * scale)
    # im_vz.set_array(vz[indices][..., i] - vz_slope[indices][..., i])
    im_vz.set_array(vz[indices][..., i])
    quiver_compression.set_UVC(strain_scale * compression_x[indices][..., i],
        strain_scale * compression_y[indices][..., i])
    quiver_extension.set_UVC(strain_scale * extension_x[indices][..., i],
        strain_scale * extension_y[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return quiver_velocity, quiver_compression, quiver_extension, txt

ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('strain_multi_filtered_vz.mp4', dpi=150)
matplotlib.pyplot.close('all')

# ---- Animate flotation ----

i = 0
fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(12, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
im = ax.imshow(flotation[indices][..., i], vmin=0, vmax=1,
    extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", "5%", pad="3%")
matplotlib.pyplot.colorbar(im, cax=cax)
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)

def update_plot(i):
    print(i)
    im.set_array(flotations[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return im, txt

ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('flotation_multi.mp4')
