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


class AttentionPooling(nn.Module):
    """ This function implements a self-attention pooling (https://arxiv.org/abs/2008.01077).

    Arguments
    ---------
    input_dim: int
        The dimension of the input Tensor
    
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
        input_dim,
        #device=None,
    ):
        super().__init__()
    
        self.input_dim = input_dim
        #self.device = device
        
        # Matmul
        self.attn_pooling_w = torch.rand((input_dim))
        
        

    def forward(self, x):
        """Returns the output the adapter.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        out = torch.matmul(x, self.attn_pooling_w)
        out = torch.nn.functional.softmax(out)
        out = torch.matmul(out.unsqueeze(1), x)

        return out.squeeze(1)
        
