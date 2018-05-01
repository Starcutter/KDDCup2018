import torch.nn.functional as F
from torch import nn
from networks import torch_utils

import torch


class CNN(nn.ModuleList):
    def __init__(
            self,
            input_shape,
            conv_layer_specs,
            activation=F.relu,
            output_activation=F.relu,
            deconv=False,
            bias=True
    ):
        for spec in conv_layer_specs:
            assert {"n_filter", "kernel_size", "pad", "stride"} <= \
                set(spec.keys())
        self.input_shape = input_shape
        self.conv_layer_specs = conv_layer_specs
        self.activation = activation
        self.output_activation = output_activation
        self.deconv = deconv

        assert len(input_shape) == 3
        input_c, input_h, input_w = input_shape

        conv_layers = []
        for i in range(len(conv_layer_specs)):
            if i == 0:
                n_input_channel = input_c
            else:
                n_input_channel = conv_layer_specs[i-1]['n_filter']
            spec = conv_layer_specs[i]
            if not 'dilation' in spec:
                spec['dilation'] = 1
            if deconv:
                link_cls = nn.ConvTranspose2d
            else:
                link_cls = nn.Conv2d
            conv_layers.append(
                link_cls(
                    in_channels=n_input_channel,
                    out_channels=spec['n_filter'],
                    kernel_size=spec['kernel_size'],
                    padding=spec['pad'],
                    stride=spec['stride'],
                    dilation=spec['dilation'],
                    bias=bias,
                )
            )
        # find out the output shape
        self.output_shape = torch_utils.get_conv_layers_output_shape(
            conv_layer_specs=conv_layer_specs,
            input_shape=(1, ) + input_shape,
            deconv=deconv,
        )[1:]
        if any([dim <= 0 for dim in self.output_shape]):
            print("Illegal output shape", self.output_shape, "for the CNN.")
            raise NotImplementedError

        super().__init__(conv_layers)

    def forward(self, state):
        h = state
        for i, layer in enumerate(self):
            h = layer(h)
            if i < len(self) - 1:
                activation = self.activation
            else:
                activation = self.output_activation
            if activation is not None:
                h = activation(h)
        return h


class FC(nn.ModuleList):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            activation=F.relu,
            output_activation=None,
            bias=True
    ):
        """

        :param input_dim:
        :param hidden_dims: a list of int
        :param output_dim:
        :param activation:
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation

        fc_layers = []
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                _input_dim = input_dim
            else:
                _input_dim = hidden_dims[i-1]
            if i == len(hidden_dims):
                _output_dim = output_dim
            else:
                _output_dim = hidden_dims[i]
            fc_layers.append(nn.Linear(_input_dim, _output_dim, bias=bias))
        super().__init__(fc_layers)

    def forward(self, state):
        h = state
        for i, layer in enumerate(self):
            h = layer(h)
            if i < len(self.hidden_dims):
                activation = self.activation
            else:
                activation = self.output_activation
            if activation is not None:
                h = activation(h)
        return h


class NNList(nn.ModuleList):
    def __init__(self, networks, name=None):
        super().__init__(networks)
        self.name = name

    def forward(self, state):
        h = state
        for nn in self:
            h = nn(h)
        return h

    def get_param_norm(self):
        return torch.norm(list(self.named_parameters())[0][1])
