"""Activations used by the FragC3 encoder."""

from torch import nn


def get_func(name):
    if name.lower() != "relu":
        raise ValueError("The released FragC3 configuration uses ReLU")
    return nn.ReLU()
