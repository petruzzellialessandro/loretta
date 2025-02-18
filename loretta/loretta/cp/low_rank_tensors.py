import torch
import math

class CPDecomposition(torch.nn.Module):
    def __init__(self, config):
        super(CPDecomposition, self).__init__()

        self.config = config
        self.factors = torch.nn.ParameterList()
        self.order = len(config.shape)
        self.rank = config.rank
        self.build_factors_Gaussian()

    def build_factors_uniform(self):
        config = self.config
        shape = config.shape
        rank = self.rank
        
        for i in range(self.order):
            n = shape[i]
            U = torch.nn.Parameter(torch.randn(n, rank) / math.sqrt(rank) * self.config.target_sdv ** (1 / self.order))
            self.factors.append(U)

    def build_factors_Gaussian(self):
        config = self.config
        shape = config.shape
        rank = self.rank
        
        for i in range(self.order):
            n = shape[i]
            U = torch.nn.Parameter(torch.randn(n, rank) / math.sqrt(rank) * self.config.target_sdv ** (1 / self.order))
            self.factors.append(U)

    def get_factors(self):
        return self.factors

    def get_full(self):
        with torch.no_grad():
            out = self.factors[0]
            for U in self.factors[1:]:
                out = torch.einsum('ir,jr->ijr', out, U)
            out = torch.sum(out, dim=-1)
        return out


class CPMatrix(torch.nn.Module):
    def __init__(self, config):
        super(CPMatrix, self).__init__()

        self.config = config
        self.factors = torch.nn.ParameterList()
        self.order = len(config.shape[0])
        self.rank = config.rank
        self.build_factors_Gaussian()

    def build_factors_uniform(self):
        config = self.config
        shape = config.shape
        rank = self.rank
        
        for i in range(self.order):
            n1 = shape[0][i]
            n2 = shape[1][i]
            U = torch.nn.Parameter(torch.randn(n1, n2, rank) / math.sqrt(rank) * self.config.target_sdv ** (1 / self.order))
            self.factors.append(U)

    def build_factors_Gaussian(self):
        config = self.config
        shape = config.shape
        rank = self.rank
        
        for i in range(self.order):
            n1 = shape[0][i]
            n2 = shape[1][i]
            U = torch.nn.Parameter(torch.randn(n1, n2, rank) / math.sqrt(rank) * self.config.target_sdv ** (1 / self.order))
            self.factors.append(U)

    def get_factors(self):
        return self.factors

    def get_full(self):
        with torch.no_grad():
            out = self.factors[0]
            for U in self.factors[1:]:
                out = torch.einsum('ijr,klr->ijklr', out, U)
            out = torch.sum(out, dim=-1)
        return out