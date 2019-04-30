import torch
import torch.nn as nn
from typing import Optional


class Normalize(nn.Module):

    """Normalize a tensor image or a batch of tensor images
    with mean and standard deviation. Input must be a tensor of shape (C, H, W)
    or a batch of tensors (*, C, H, W).
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.*Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (Tensor): Mean for each channel.
        std (Tensor): Standard deviation for each channel.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:

        super(Normalize, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self):
        repr = '(mean={0}, std={1})'.format(self.mean, self.std)
        return self.__class__.__name__ + repr


def normalize(data: torch.Tensor, mean: torch.Tensor,
              std: torch.Tensor) -> torch.Tensor:

    """Normalise the image with channel-wise mean and standard deviation.

    Args:
        data (Tensor): The image tensor to be normalised.
        mean (Tensor): Mean for each channel.
        std (Tensor): Standard deviations for each channel.

    Returns:
        Tensor: The normalised image tensor.
    """

    if not torch.is_tensor(data):
        raise TypeError('data should be a tensor. Got {}'.format(type(data)))

    if not torch.is_tensor(mean):
        raise TypeError('mean should be a tensor. Got {}'.format(type(mean)))

    if not torch.is_tensor(std):
        raise TypeError('std should be a tensor. Got {}'.format(type(std)))

    if len(mean) != data.shape[-3]:
        raise ValueError('mean lenght and number of channels do not match')

    if len(std) != data.shape[-3]:
        raise ValueError('std lenght and number of channels do not match')

    mean = mean[..., :, None, None].to(data.device)
    std = std[..., :, None, None].to(data.device)

    out = data.sub(mean).div(std)

    return out

# - denormalise