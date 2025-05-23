from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import linox
import linox.utils
from jax import numpy as jnp
from jax.typing import ArrayLike
from linox.typing import LinearOperatorLike

from luno.jacobians.fno import LastFNOBlockWeightJacobian
from luno.models.fno import FFTGrid, fno_block
from luno.randprocs import ParametricGaussianProcess


class FNOGPLastLayer:
    def __init__(
        self,
        fno_head: Callable[[jax.Array], jax.Array],
        R: ArrayLike,
        W: ArrayLike,
        b: ArrayLike,
        weight_cov: LinearOperatorLike,
        projection: Callable[[jax.Array], jax.Array],
        num_output_channels: int,
    ) -> None:
        self._fno_head = fno_head

        self._R = jnp.asarray(R)
        self._W = jnp.asarray(W)
        self._b = jnp.asarray(b)

        self._weight_cov = linox.utils.as_linop(weight_cov)

        self._projection = projection
        self._num_output_channels = num_output_channels

    def __call__(self, a: ArrayLike) -> FNOGPLastLayer.FixedInputGaussianProcess:
        a = jnp.asarray(a)

        v_in = self._fno_head(a)

        return FNOGPLastLayer.FixedInputGaussianProcess(
            v_in,
            self._R,
            self._W,
            self._b,
            self._weight_cov,
            self._projection,
            self._num_output_channels,
        )

    class FixedInputGaussianProcess(ParametricGaussianProcess):
        def __init__(
            self,
            v_in: jax.Array,
            R: jax.Array,
            W: jax.Array,
            b: jax.Array,
            weight_cov: linox.LinearOperator,
            projection: Callable[[jax.Array], jax.Array],
            num_output_channels: int,
        ):
            self._v_in = v_in

            self._R = R
            self._W = W
            self._b = b

            self._projection = projection
            self._num_output_channels = num_output_channels

            super().__init__(weight_cov=weight_cov)

        @functools.singledispatchmethod
        def mean_and_features(
            self, x: ArrayLike, /
        ) -> tuple[jax.Array, linox.LinearOperator]:
            raise NotImplementedError()

        @mean_and_features.register
        def _(self, x: FFTGrid, /) -> tuple[jax.Array, linox.LinearOperator]:
            v_out, intermediates = fno_block(
                self._v_in,
                R=self._R,
                W=self._W,
                b=self._b,
                output_grid_shape=x.shape[:-1],
            )

            u = self._projection(v_out)

            features = LastFNOBlockWeightJacobian(
                v_in=self._v_in,
                R=self._R,
                W=self._W,
                b=self._b,
                output_grid_shape=x.shape[:-1],
                projection=self._projection,
                num_output_channels=self._num_output_channels,
                z_in=intermediates["spectral_convolution"]["z_in"],
                v_out=v_out,
                dtype=x.dtype,  # Infer dtype from grid.
            )

            return u.reshape(-1), features
