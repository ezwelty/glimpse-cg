import cg
from cg import glimpse
from glimpse.imports import (np, re, copy)

# ---- Functions: Statistics ----

def numpy_dropdims(a, axis=None, keepdims=False):
    a = np.asarray(a)
    if keepdims:
        return a
    elif a.size == 1:
        return np.asscalar(a)
    elif axis is not None and a.shape[axis] == 1:
        return a.squeeze(axis=axis)
    else:
        return a

def prepare_normals(means, sigmas, weights, normalize, axis):
    isnan_mean = np.isnan(means)
    isnan_sigmas = np.isnan(sigmas)
    if np.any(isnan_mean != isnan_sigmas):
        raise ValueError('mean and sigma NaNs do not match')
    if np.any(sigmas == 0):
        raise ValueError('sigmas cannot be 0')
    if weights is None:
        weights = np.ones(means.shape)
    if normalize:
        weights = weights * (1 / np.nansum(weights * ~isnan_mean, axis=axis, keepdims=True))
    return isnan_mean, isnan_sigmas, weights

def sum_normals(means, sigmas, weights=None, normalize=False, correlation=0,
    axis=None, keepdims=False):
    """
    Return the mean and sigma of the sum of random variables.

    See https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Linear_combinations.

    Arguments:
        means (numpy.ndarray): Variable means
        sigmas (numpy.ndarray): Variable standard deviations
        weights (numpy.ndarray): Variable weights
        normalize (bool): Whether to normalize weights so that they sum to 1
            for non-missing values
        correlation (float): Correlation to assume between pairs of different variables
    """
    isnan_mean, isnan_sigmas, weights = prepare_normals(
        means, sigmas, weights, normalize, axis)
    wmeans = np.nansum(weights * means, axis=axis, keepdims=True)
    # Initialize variance as sum of diagonal elements
    variances = np.nansum(weights**2 * sigmas**2, axis=axis, keepdims=True)
    # np.nansum interprets sum of nans as 0
    allnan = isnan_mean.all(axis=axis, keepdims=True)
    wmeans[allnan] = np.nan
    variances[allnan] = np.nan
    if correlation:
        # Add off-diagonal elements
        n = means.size if axis is None else means.shape[axis]
        pairs = np.triu_indices(n=n, k=1)
        variances += 2 * np.nansum(correlation *
            np.take(weights, pairs[0], axis=axis) *
            np.take(weights, pairs[1], axis=axis) *
            np.take(sigmas, pairs[0], axis=axis) *
            np.take(sigmas, pairs[1], axis=axis), axis=axis, keepdims=True)
    return (
        numpy_dropdims(wmeans, axis=axis, keepdims=keepdims),
        numpy_dropdims(np.sqrt(variances), axis=axis, keepdims=keepdims))

def mix_normals(means, sigmas, weights=None, normalize=False,
    axis=None, keepdims=False):
    """
    Return the mixture distribution of a sum of random variables.

    See https://en.wikipedia.org/wiki/Mixture_distribution#Moments.

    Arguments:
        means (numpy.ndarray): Variable means
        sigmas (numpy.ndarray): Variable standard deviations
        weights (numpy.ndarray): Variable weights
        normalize (bool): Whether to normalize weights so that they sum to 1
            for non-missing values
    """
    isnan_mean, isnan_sigmas, weights = prepare_normals(
        means, sigmas, weights, normalize, axis)
    wmeans = np.nansum(weights * means, axis=axis, keepdims=True)
    variances = np.nansum(weights * (means**2 + sigmas**2), axis=axis, keepdims=True) - wmeans**2
    # np.nansum interprets sum of nans as 0
    isnan = isnan_mean.all(axis=axis, keepdims=True)
    wmeans[isnan] = np.nan
    variances[isnan] = np.nan
    return (
        numpy_dropdims(wmeans, axis=axis, keepdims=keepdims),
        numpy_dropdims(np.sqrt(variances), axis=axis, keepdims=keepdims))

# ---- Functions: Tracks operations ----

def reverse_tracks(tracks):
    """
    Return Tracks in reverse temporal order.
    """
    new = copy.copy(tracks)
    new.datetimes = tracks.datetimes[::-1]
    new.means = tracks.means[:, ::-1, ...]
    new.sigmas = None if tracks.sigmas is None else tracks.sigmas[:, ::-1, ...]
    new.covariances = None if tracks.covariances is None else tracks.covariances[:, ::-1, ...]
    new.particles = None if tracks.particles is None else tracks.particles[:, ::-1, ...]
    new.weights = None if tracks.weights is None else tracks.weights[:, ::-1, ...]
    return new

def mean_tracks(tracks):
    means, sigmas = sum_normals(
        means=tracks.means, sigmas=tracks.sigmas,
        weights=tracks.sigmas**-2, normalize=True, correlation=1, axis=1, keepdims=True)
    return glimpse.Tracks(
        tracks.datetimes[[0, -1]],
        np.tile(means, (1, 2, 1)), np.tile(sigmas, (1, 2, 1)))

def select_repeat_tracks(runs):
    """
    Return Tracks composed of the best track for each initial point.

    Selects the track for each point that minimizes the temporal mean standard
    deviation for vx + vy, i.e. mean(sqrt(vx_sigma**2 + vy_sigma**2)).

    Arguments:
        runs (iterable): Tracks objects with identical point and time dimensions
    """
    # Compute metric for each track
    metric = np.row_stack([
        np.nanmean(np.sqrt(np.nansum(
            run.sigmas[..., 3:5]**2, axis=2)), axis=1)
        for run in runs])
    # Choose run with the smallest metric
    selected = np.argmin(metric, axis=0)
    # Merge runs
    means = runs[0].means.copy()
    sigmas = runs[0].sigmas.copy()
    for i, run in enumerate(runs[1:], start=1):
        mask = selected == i
        means[mask, ...] = run.means[mask, ...]
        sigmas[mask, ...] = run.sigmas[mask, ...]
    return glimpse.Tracks(
        datetimes=runs[0].datetimes, means=means, sigmas=sigmas)

def merge_repeat_tracks(runs):
    means = np.stack([run.means for run in runs], axis=3)
    sigmas = np.stack([run.sigmas for run in runs], axis=3)
    means, sigmas = mix_normals(
        means=means, sigmas=sigmas,
        weights=sigmas**-2, normalize=True, axis=3)
    return glimpse.Tracks(runs[0].datetimes, means=means, sigmas=sigmas)

def compute_principal_strains(vx, vy, d):
    """
    Compute principal strain rates from velocity fields.

    Arguments:
        vx (array-like): Velocity along x (x, y, t)
        vy (array-like): Velocity along y (x, y, t)
        d (iterable): Velocity field grid size in the same distance units as
            the velocities in vx and vy (x, y)

    Returns:
        numpy.ndarray: Principal extension along x (x, y, t)
        numpy.ndarray: Principal extension along y (x, y, t)
        numpy.ndarray: Principal compression along x (x, y, t)
        numpy.ndarray: Principal compression along y (x, y, t)
    """
    ndims = vx.ndim
    vx = np.atleast_3d(vx)
    vy = np.atleast_3d(vy)
    dudy, dudx = np.gradient(vx, axis=(0, 1))
    dudx, dudy = dudx * (1 / d[0]), dudy * (1 / d[1])
    dvdy, dvdx = np.gradient(vy, axis=(0, 1))
    dvdx, dvdy = dvdx * (1 / d[0]), dvdy * (1 / d[1])
    strain = (dudx, dvdy, dudy + dvdx)
    theta = np.arctan2(strain[2], (strain[0] - strain[1])) * 0.5
    cos, sin = np.cos(theta), np.sin(theta)
    Q = np.stack([cos.ravel(), sin.ravel(), -sin.ravel(), cos.ravel()]).reshape(2, 2, -1)
    E = np.stack([strain[0].ravel(), strain[2].ravel() * 0.5, strain[2].ravel() * 0.5, strain[1].ravel()]).reshape(2, 2, -1)
    E_prime = np.diagonal(np.matmul(Q.transpose(2, 0, 1),
        np.matmul(E.transpose(2, 0, 1), Q.transpose(1, 0, 2).transpose(2, 0, 1))
        ).transpose(1, 2, 0))
    emax = E_prime[:, 0].reshape(theta.shape)
    emin = E_prime[:, 1].reshape(theta.shape)
    theta_rot = theta + np.pi * 0.5
    extension_u = emax * np.cos(theta)
    extension_v = emax * np.sin(theta)
    compression_u = emin * np.cos(theta_rot)
    compression_v = emin * np.sin(theta_rot)
    if ndims == 2:
        return (np.squeeze(extension_u, axis=2), np.squeeze(extension_v, axis=2),
        np.squeeze(compression_u, axis=2), np.squeeze(compression_v, axis=2))
    else:
        return extension_u, extension_v, compression_u, compression_v

# ---- Functions: Flatten tracks ----

def flatten_tracks_ethan(runs):
    # Select best of repeat runs
    f = select_repeat_tracks((runs['f'], runs['fv']))
    r = select_repeat_tracks((runs['r'], runs['rv']))
    # Reverse reverse run
    rr = reverse_tracks(r)
    # Merge forward/backward runs
    # Computes mixture distribution of inverse-variance weighted sum
    means = np.stack((f.means[..., 3:], rr.means[..., 3:]), axis=3)
    sigmas = np.stack((f.sigmas[..., 3:], rr.sigmas[..., 3:]), axis=3)
    means, sigmas = mix_normals(
        means=means, sigmas=sigmas, weights=sigmas**-2, normalize=True, axis=3)
    # Flatten merged run
    # Computes mean/variance of inverse-variance weighted linear combination
    # (assumes correlation = 1 for all variable pairs)
    return sum_normals(means=means, sigmas=sigmas, weights=sigmas**-2,
        normalize=True, correlation=1, axis=1)

def flatten_tracks_doug(runs):
    # Join together second forward and backward runs
    f, r = runs['fv'], runs['rv']
    means = np.column_stack((f.means[..., 3:], r.means[..., 3:]))
    sigmas = np.column_stack((f.sigmas[..., 3:], r.sigmas[..., 3:]))
    # Flatten joined runs
    # Mean: Inverse-variance weighted mean
    # Sigma: Linear combination of weighted correlated random variables
    # (approximation using the weighted mean of the variances)
    weights = sigmas**-2
    weights *= 1 / np.nansum(weights, axis=1, keepdims=True)
    allnan = np.isnan(means).all(axis=1, keepdims=True)
    means = np.nansum(weights * means, axis=1, keepdims=True)
    sigmas = np.sqrt(np.nansum(weights * sigmas**2, axis=1, keepdims=True))
    # np.nansum interprets sum of nans as 0
    means[allnan] = np.nan
    sigmas[allnan] = np.nan
    return means.squeeze(axis=1), sigmas.squeeze(axis=1)
    # return sum_normals(means=means, sigmas=sigmas, weights=sigmas**-2,
    #     normalize=True, correlation=1, axis=1)
