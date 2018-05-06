import torch
import torch.optim as optim
import math
import numpy as np
import torch.nn.functional as F
from torch import nn


def get_variable(np_array, cuda=False, **kwargs):
    assert isinstance(np_array, np.ndarray)
    var = torch.autograd.Variable(torch.from_numpy(np_array), **kwargs)
    if np_array.dtype in [np.float16, np.float32, np.float64]:
        var = var.type(torch.FloatTensor)
    if cuda:
        var = var.cuda()
    return var


def to_numpy(input):
    if isinstance(input, torch.autograd.Variable):
        tensor = input.data
    else:
        tensor = input
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def get_constant_tensor(shape, value, dtype, cuda=False):
    if cuda:
        if dtype == "float":
            tensor = torch.cuda.FloatTensor(*shape)
        elif dtype == "byte":
            tensor = torch.cuda.ByteTensor(*shape)
        else:
            raise NotImplementedError
    else:
        if dtype == "float":
            tensor = torch.cuda.FloatTensor(*shape)
        elif dtype == "byte":
            tensor = torch.ByteTensor(*shape)
        else:
            raise NotImplementedError
    tensor.fill_(value)
    return tensor


def get_modules_output_shape(modules, input_shape):
    input_var = get_variable(np.zeros(input_shape).astype(np.float32))
    output_var = input_var
    for m in modules:
        output_var = m(output_var)
    return output_var.data.shape


def get_conv_layers_output_shape(conv_layer_specs, input_shape, deconv=False):
    if deconv:
        raise NotImplementedError
    assert len(input_shape) == 4  # N, C, H, W
    assert input_shape[2] == input_shape[3], "only supports square inputs now"
    size = input_shape[2]
    for spec in conv_layer_specs:
        size = (size + 2 * spec['pad'] - spec['dilation'] *
                (spec['kernel_size'] - 1) - 1) // spec['stride'] + 1
    n_output_channel = conv_layer_specs[-1]['n_filter']
    output_shape = (input_shape[0], n_output_channel, size, size)
    return output_shape


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Average(nn.Module):
    def __init__(self, dim=None):
        super(Average, self).__init__()
        self.dim = dim

    def forward(self, input):
        if self.dim is None:
            return torch.mean(input)
        return torch.mean(input, self.dim, False)


class Sum(nn.Module):
    def __init__(self, dim=None):
        super(Sum, self).__init__()
        self.dim = dim

    def forward(self, input):
        if self.dim is None:
            return torch.sum(input)
        return torch.sum(input, self.dim, False)
