import math
import torch
import numpy as np
import torch.nn as nn
from .low_rank_tensors import CPDecomposition, CPMatrix  # Replace TensorTrain with CPDecomposition
from .utils import config_class, quantize, CP_forward_quant  # Replace TT_forward_quant with CP_forward_quant


class wrapped_linear_layers(nn.Module):
    def __init__(self, in_features, out_features, bias=True, tensorized=False, config=None):
        super(wrapped_linear_layers, self).__init__()
        if tensorized:
            self.layer = TensorizedLinear_module(in_features, out_features, config, bias=bias)
        else:
            self.layer = torch.nn.Linear(in_features, out_features, bias=bias)

        self.tensorized = tensorized

    def forward(self, input, config_forward=None):
        if self.tensorized:
            return self.layer(input, config_forward=config_forward)
        else:
            return self.layer(input)


class TensorizedLinear_module(nn.Module):
    def __init__(self, in_features, out_features, config, bias=True):
        """
        config has the following attributes:
        shape: the shape of the tensor
        rank: the rank of the CP decomposition
        set_scale_factors: True or False
        """
        super(TensorizedLinear_module, self).__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(1 / (self.in_features + self.out_features))
        config_tensor = config_class(shape=config.shape, rank=config.ranks, target_sdv=target_stddev)
        self.tensor = CPDecomposition(config_tensor)  # Use CPDecomposition instead of TensorTrain
        self.tensor_shape = config.shape

        if bias == False:
            self.bias = 0
        else:
            stdv = 1. / math.sqrt(out_features)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

        if hasattr(config, 'set_scale_factors') and config.set_scale_factors == True:
            self.set_scale_factors()

    def set_scale_factors(self, scale_w=1.0, scale_input=1.0, scale_intermediate=1.0, scale_dy=1.0, scale_x=1.0, scale_out=1.0):
        self.scales = torch.nn.ParameterList()
        self.scale_factors = torch.nn.ParameterList()

        if not isinstance(scale_w, list):
            scale_w = [scale_w] * self.tensor.order
        for s in scale_w:
            self.scale_factors.append(torch.nn.Parameter(torch.tensor(s)))

        self.scale_input = torch.nn.Parameter(torch.tensor(scale_input))
        self.scale_intermediate = torch.nn.Parameter(torch.tensor(scale_intermediate))
        self.scale_dy = torch.nn.Parameter(torch.tensor(scale_dy))
        self.scale_x = torch.nn.Parameter(torch.tensor(scale_x))
        self.scale_out = torch.nn.Parameter(torch.tensor(scale_out))

        self.scales.append(self.scale_input)
        self.scales.append(self.scale_intermediate)
        self.scales.append(self.scale_dy)
        self.scales.append(self.scale_x)
        self.scales.append(self.scale_out)

    def forward(self, input, config_forward=None):
        """
        config_forward:
        prune_mask: True or False. Use prune mask or not
        threshold: float number. The threshold to clip rank_parameters to 0
        quantized: 0: full precision. 1: quantization-aware training. 2: low-precision training.
        if quantized:
            rep: INT or FLOAT. quantization type
            bit_input/factors/intermediate/out: bits for each part
            rounding: stochastic or nearest. Rounding type
        """
        if config_forward is None:
            factors = self.tensor.get_factors()
        else:
            factors = self.tensor.get_factors()

        if config_forward is None or config_forward.quantized == 0:
            out = self.forward_cp_full_precision(input, factors) + self.bias

        elif config_forward.quantized == 1:
            out = self.forward_cp_quantization_aware(input, factors, config_forward) + self.bias

        elif config_forward.quantized == 2:
            input = quantize.apply(input, self.scale_input, config_forward.bit_input, config_forward.rep, config_forward.rounding)
            Q_factors = []
            for i, U in enumerate(factors):
                Q_factors.append(quantize.apply(U, self.scale_factors[i], config_forward.bit_factors, config_forward.rep, config_forward.rounding))
            out = CP_forward_quant.apply(config_forward.rounding, config_forward.bit_intermediate, self.scale_intermediate, self.scale_dy, self.scale_dy, input, *Q_factors).clone()
            out = out + self.bias

        return out

    def forward_cp_quantization_aware(self, input, factors, config_forward):
        input = quantize.apply(input, self.scale_input, config_forward.bit_input, config_forward.rep, config_forward.rounding)
        Q_factors = []
        for i, U in enumerate(factors):
            Q_factors.append(quantize.apply(U, self.scale_factors[i], config_forward.bit_factors, config_forward.rep, config_forward.rounding))
        factors = Q_factors

        quant_intermediate = lambda x: quantize.apply(x, self.scale_intermediate, config_forward.bit_intermediate, config_forward.rep, config_forward.rounding)
        quant_x = lambda x: quantize.apply(x, self.scale_x, config_forward.bit_intermediate, config_forward.rep, config_forward.rounding)
        quant_out = lambda x: quantize.apply(x, self.scale_out, config_forward.bit_out, config_forward.rep, config_forward.rounding)

        # CP decomposition forward pass
        out = factors[0]
        for i in range(1, len(factors)):
            out = quant_intermediate(torch.einsum('ir,jr->ijr', out, factors[i]))
        out = quant_x(torch.sum(out, dim=-1))

        return out

    def forward_cp_full_precision(self, input_mat, factors):
        # CP decomposition forward pass
        # Compute the reconstructed weight matrix using einsum
        # Dynamically generate einsum equation based on the number of factors
        num_factors = len(factors)
        input_subscripts = ','.join([f'{chr(97 + i)}r' for i in range(num_factors)])
        output_subscript = ''.join([chr(97 + i) for i in range(num_factors)])
        equation = f'{input_subscripts}->{output_subscript}'
        
        # Compute the full tensor using einsum and reshape to a matrix
        reconstructed_tensor = torch.einsum(equation, *factors)
        reconstructed_matrix = reconstructed_tensor.reshape(self.in_features, self.out_features)
        
        # Perform the matrix multiplication
        output = input_mat @ reconstructed_matrix
        return output