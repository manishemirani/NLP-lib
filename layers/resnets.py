from typing import Callable

import ml_collections

from .highway import Highway
from .actnorm import ActNorm

import flax.linen as nn


class ResNet(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, y):

        if self.config.res_operation == 'highway':
            assert self.config.residual_features and self.config.residual_layre != None, (
                'features and num layers are None values!')
            highway = Highway(
                features=self.config.residual_features,
                num_layers=self.confdig.residual_layre,
                use_bias=self.config.residual_use_bias,
                use_y=True
            )

            return highway((x, y))

        if self.config.res_operation == 'add':
            return x + y


class IResnet(nn.Module):
    func: Callable
    mode: str = 'inverse'
    max_iter: int = 100
    use_actnorm: bool = True

    @nn.compact
    def __call__(self, y):
        x = y
        for _ in range(self.max_iter):
            x = y - self.func(x, x)

        if self.use_actnorm:
            actnorm = ActNorm(x.shape)
            x = actnorm(x, self.mode)

        return x
