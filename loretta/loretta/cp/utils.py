import torch
import numpy as np
import torch.nn.functional as F

class config_class():
    def __init__(self, **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))

class quantize(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    pass

class CP_forward_quant(torch.autograd.Function):
    pass

class CP_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, *factors):
        """
        Forward pass for CP decomposition.
        """
        with torch.no_grad():
            # Get the shape of the input matrix and factors
            ctx.input_shape = matrix.shape
            ctx.factors = factors
            ctx.matrix = matrix

            # Compute the output using CP decomposition
            output = factors[0]
            for i in range(1, len(factors)):
                output = torch.einsum('ir,jr->ijr', output, factors[i])
            output = torch.sum(output, dim=-1)  # Sum over the rank dimension

            # Reshape the output to match the expected shape
            out_shape = [matrix.shape[0], np.prod([f.shape[1] for f in factors])]
            output = output.reshape(out_shape)

            # Save tensors for backward pass
            ctx.save_for_backward(matrix, *factors)

        return output

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass for CP decomposition.
        """
        with torch.no_grad():
            matrix, *factors = ctx.saved_tensors

            # Compute gradients for each factor
            grads = []
            for i, factor in enumerate(factors):
                # Compute the gradient for the i-th factor
                grad = torch.einsum('ij,jk->ik', matrix, dy)
                for j, f in enumerate(factors):
                    if j != i:
                        grad = torch.einsum('ik,jk->ij', grad, f)
                grads.append(grad)

            # Compute the gradient for the input matrix
            dx = torch.einsum('ij,jk->ik', dy, factors[0])
            for f in factors[1:]:
                dx = torch.einsum('ij,jk->ik', dx, f)

            # Reshape the gradient to match the input shape
            dx = dx.reshape(ctx.input_shape)

        return dx, *grads