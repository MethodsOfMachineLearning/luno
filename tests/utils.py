import numpy as np
import scipy.stats

from numpy.typing import ArrayLike


def assert_samples_marginally_gaussian(
    samples: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    axis: int = 0,
):
    samples = np.sort(samples, axis=axis)

    mean = np.expand_dims(mean, axis=axis)
    std = np.expand_dims(std, axis=axis)

    samples_broadcast = np.broadcast_to(samples, shape=samples.shape)
    mean_broadcast = np.broadcast_to(mean, shape=samples.shape)
    std_broadcast = np.broadcast_to(std, shape=samples.shape)

    nondegenerate = std_broadcast > 0

    # Zero-variance marginals are deterministic and should match the mean.
    if np.any(~nondegenerate):
        np.testing.assert_allclose(
            samples_broadcast[~nondegenerate],
            mean_broadcast[~nondegenerate],
            atol=1e-6,
        )

    if not np.any(nondegenerate):
        return

    samples_standardized = np.zeros_like(samples_broadcast, dtype=np.float64)
    samples_standardized[nondegenerate] = (
        samples_broadcast[nondegenerate] - mean_broadcast[nondegenerate]
    ) / std_broadcast[nondegenerate]

    # Map standardized samples through standard normal cdf and compare to uniform cdf
    samples_norm_cdf = scipy.stats.norm.cdf(samples_standardized)
    uniform_cdf = np.broadcast_to(
        np.moveaxis(
            np.expand_dims(
                np.linspace(0.0, 1.0, samples_norm_cdf.shape[axis]),
                axis=tuple(range(1, samples_norm_cdf.ndim)),
            ),
            0,
            axis,
        ),
        shape=samples_norm_cdf.shape,
    )

    np.testing.assert_allclose(
        samples_norm_cdf[nondegenerate],
        uniform_cdf[nondegenerate],
        atol=6e-2,
    )
