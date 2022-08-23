"""Library implementing adapters.

Authors
 * Ha Nguyen 2022
"""

import torch
import torch.nn as nn
import logging
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm

logger = logging.getLogger(__name__)


class SerialAdapter(nn.Module):
    """ This function implements a complex-valued LSTM.

    Input format is (batch, time, fea) or (batch, time, fea, channel).
    In the latter shape, the two last dimensions will be merged:
    (batch, time, fea * channel)

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_dim: int
        The dimension of the input Tensor
    bottleneck_dim: int
        The dimension of the bottleneck
    bias : bool
        If True, the additive bias b is adopted.
    activation: str 
        The type of activation function (default: ReLU)

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 40])
    >>> rnn = CLSTM(hidden_size=16, input_shape=inp_tensor.shape)
    >>> out_tensor = rnn(inp_tensor)
    >>>
    torch.Size([10, 16, 32])
    """

    def __init__(
        self,
        n_neurons,
        input_dim=None,
        bottleneck_dim=None,
        bias=True,
        activation="ReLU",
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        
        if input_dim is None:
            input_dim = n_neurons
        if bottleneck_dim is None:
            bottleneck_dim = n_neurons
        
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.bias = bias
        self.activation = activation

        if activation == "ReLU":
            self.activation_fuction = nn.ReLU()
        
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if input_dim != n_neurons:
            # when the input's and output's dimensions are different,
            # we need to project the input's dimension to the output's dimension
            self.proj = nn.Linear(input_dim, n_neurons, bias=bias)
            self.proj_activation = nn.ReLU()
        
        self.linear1 = nn.Linear(n_neurons, bottleneck_dim, bias=bias)
        self.linear2 = nn.Linear(bottleneck_dim, n_neurons, bias=bias)
        self.layernorm = nn.LayerNorm(n_neurons, eps=eps, elementwise_affine=elementwise_affine)
        

    def forward(self, x):
        """Returns the output the adapter.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """

        #if self.proj:
        if hasattr(self, 'proj'):
            x = self.proj(x)
            x = self.proj_activation(x)
        
        out = self.layernorm(x)
        out = self.linear1(out)
        out = self.activation_fuction(out)
        out = self.linear2(out)
        # residual connection
        out += x
        
        return out

class LengthAdapter(nn.Module):
    """A lightweight adaptor module in between encoder and decoder 
    to performs projection and downsampling to reduce the 
    length discrepancy between the input and output sequences.
    Original idea can be found in here:
    https://arxiv.org/pdf/2010.12829.pdf
    
    Arguments
    ---------
    n_neurons: int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_dim: int
        The dimension of the input Tensor
    n_layers: int
        The number of convolution blocks
    kernel_size: int (default: 3)
        The kernel size of each convolution block
    stride: int (default: 2)
        The stride of each convolution block
    dilation: int (default: 1)
        The dilation of each convolution block
    norm: bool (default: True)
        Whether or not to apply batch normalization after each conv block
    bias: bool (default: True)
        If True, the additive bias b is adopted
    """
    def __init__(
        self,
        n_neurons,
        input_dim=None,
        n_layers=3,
        kernel_size=3,
        stride=2,
        dilation=1,
        norm=None,
        eps=1e-05,
        affine=True,
        bias=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = norm 
        
        from collections import OrderedDict
        modules = OrderedDict()
        for i in range(n_layers):
            conv = nn.Conv1d(
                    input_dim, 
                    n_neurons, 
                    kernel_size=kernel_size, 
                    stride=stride, 
                    dilation=dilation,
                    bias=bias,
            )
            modules['Conv1D_' + str(i+1)] = conv
            if norm is not None:
                batchnorm = nn.BatchNorm1d(
                        n_neurons,
                        eps=eps,
                        affine=affine
                )
                modules['BatchNorm1D_' + str(i+1)] = batchnorm
            input_dim = n_neurons
        self.model = nn.Sequential(modules)

    def forward(self, x):
        """Returns the output the adapter.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape (batch size, length, feature dim).
        ---------
        Return Tensor of shape (batch size, length, feature dim).
        """

        return self.model(x.transpose(1, 2)).transpose(1, 2)



