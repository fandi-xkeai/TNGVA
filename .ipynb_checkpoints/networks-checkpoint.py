'''
***Closely based on code for TVAE by Matthew J. Vowels***
https://github.com/matthewvowels1/TVAE_release
See also the paper:
Vowels, M. J., Camgoz, N. C., & Bowden, R. (2021). Targeted VAE: Variational 
and targeted learning for causal inference. https://arxiv.org/pdf/2009.13472
'''
import torch
import torch.nn as nn
import pyro.distributions as dist
from pygcn.layers import GraphConvolution
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE_residual(nn.Module):
    def __init__(self, in_features, out_features, dropout=0, residual_weight=0.001, aggr='sum'):
        super(GraphSAGE_residual, self).__init__()
        self.sage = SAGEConv(in_features, out_features, aggr=aggr) 
        self.residual = nn.Linear(in_features, out_features) 
        self.dropout = dropout
        self.residual_weight = residual_weight  

    def forward(self, x, edge_index):
        h_sage = self.sage(x, edge_index) 
        h_residual = self.residual(x)  
        h_out = self.residual_weight * h_sage + (1-self.residual_weight) * h_residual
        h_out = F.dropout(h_out, self.dropout, training=self.training)  
        return h_out


class GraphSAGENet(nn.Module):
    def __init__(self, sizes, dropout=0, final_activation=None):
        super(GraphSAGENet, self).__init__()
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(SAGEConv(in_size, out_size, aggr='mean'))  
            layers.append(nn.ReLU()) 
        layers.pop(-1)  
        if final_activation is not None:
            layers.append(final_activation)  
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        if x.dim() == 3:  
            num_samples = x.shape[0]
            hidden = []
            for i in range(num_samples):
                h = x[i]
                for layer in self.layers:
                    if isinstance(layer, SAGEConv):
                        h = layer(h, edge_index)  
                    else:
                        h = layer(h)  
                hidden.append(h)
            return torch.stack(hidden, dim=0)
        else: 
            for layer in self.layers:
                if isinstance(layer, SAGEConv):
                    x = layer(x, edge_index)  
                else:
                    x = layer(x)  
            return x


class GraphSAGENet_residual(nn.Module):  
    def __init__(self, sizes, dropout=0, final_activation=None):
        super(GraphSAGENet_residual, self).__init__()
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(GraphSAGE_residual(in_size, out_size, dropout=dropout))  
            layers.append(nn.ReLU()) 
        layers.pop(-1) 
        if final_activation is not None:
            layers.append(final_activation)  
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        if x.dim() == 3:  
            num_samples = x.shape[0]
            hidden = []
            for i in range(num_samples):
                h = x[i]
                for layer in self.layers:
                    if isinstance(layer, GraphSAGE_residual):
                        h = layer(h, edge_index)  
                    else:
                        h = layer(h)  
                hidden.append(h)
            return torch.stack(hidden, dim=0)
        else: 
            for layer in self.layers:
                if isinstance(layer, GraphSAGE_residual):
                    x = layer(x, edge_index)  
                else:
                    x = layer(x)  
            return x


class GraphSAGENet_BernoulliNet_residual(nn.Module):  
    def __init__(self, input_dim, hidden_dim):
        super(GraphSAGENet_BernoulliNet_residual, self).__init__()
        self.sage1 = GraphSAGE_residual(input_dim, hidden_dim, dropout=0)
        self.elu = nn.ReLU()
        self.sage2 = GraphSAGE_residual(hidden_dim, 1, dropout=0)

    def forward(self, x, edge_index):
        if x.dim() == 3:  
            num_samples = x.shape[0]
            logits_list = []
            for i in range(num_samples):
                h = self.sage1(x[i], edge_index)
                h = self.elu(h)
                logits = self.sage2(h, edge_index)
                logits = logits.squeeze(-1).clamp(min=-10, max=10)
                logits_list.append(logits)
            logits = torch.stack(logits_list, dim=0)
        else:  
            h = self.sage1(x, edge_index)
            h = self.elu(h)
            logits = self.sage2(h, edge_index)
            logits = logits.squeeze(-1).clamp(min=-10, max=10)
        return logits,

    @staticmethod
    def make_dist(logits):
        return torch.distributions.Bernoulli(logits=logits)

class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)


class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """
    @staticmethod
    def get_class(dtype):
        """
        Get a subclass by a prefix of its name, e.g.::

            assert DistributionNet.get_class("bernoulli") is BernoulliNet
        """
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))


class BernoulliNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return logits,

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class ExponentialNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``rate``.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = ExponentialNet([3, 4])
        x = torch.randn(3)
        rate, = net(x)
        y = net.make_dist(rate).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        scale = nn.functional.softplus(self.fc(x).squeeze(-1)).clamp(min=1e-3, max=1e6)
        rate = scale.reciprocal()
        return rate,

    @staticmethod
    def make_dist(rate):
        return dist.Exponential(rate)


class LaplaceNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Laplace random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = LaplaceNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Laplace(loc, scale)


class NormalNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)

  
    
class StudentTNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``df,loc,scale``
    triple, with shared ``df > 1``.

    This is used to represent a conditional probability distribution of a
    single Student's t random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = StudentTNet([3, 4])
        x = torch.randn(3)
        df, loc, scale = net(x)
        y = net.make_dist(df, loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])
        self.df_unconstrained = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)
        return df, loc, scale

    @staticmethod
    def make_dist(df, loc, scale):
        return dist.StudentT(df, loc, scale)


class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()

    This is intended for the latent ``z`` distribution and the prewhitened
    ``x`` features, and conservatively clips ``loc`` and ``scale`` values.
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale
    
    
class DiagGraphSAGENet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiagGraphSAGENet, self).__init__()
        self.sage1 = SAGEConv(in_dim, out_dim, aggr='sum')
        self.sage2 = SAGEConv(in_dim, out_dim, aggr='sum')

    def forward(self, x, edge_index):
        if x.dim() == 3:
            num_samples = x.shape[0]
            hidden = []
            for i in range(num_samples):
                xi = x[i]
                loc = self.sage1(xi, edge_index).clamp(min=-1e2, max=1e2)
                scale = F.softplus(self.sage2(xi, edge_index)).add(1e-3).clamp(max=1e2)
                hidden.append((loc, scale))
            loc, scale = zip(*hidden)
            loc = torch.stack(loc, dim=0)
            scale = torch.stack(scale, dim=0)
        else:
            loc = self.sage1(x, edge_index).clamp(min=-1e2, max=1e2)
            scale = F.softplus(self.sage2(x, edge_index)).add(1e-3).clamp(max=1e2)
        return loc, scale

class DiagGraphSAGENet_residual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DiagGraphSAGENet_residual, self).__init__()
        self.sage1 = GraphSAGE_residual(in_dim, out_dim, dropout=0)
        self.sage2 = GraphSAGE_residual(in_dim, out_dim, dropout=0)

    def forward(self, x, edge_index):
        if x.dim() == 3:
            num_samples = x.shape[0]
            hidden = []
            for i in range(num_samples):
                xi = x[i]
                loc = self.sage1(xi, edge_index).clamp(min=-1e2, max=1e2)
                scale = F.softplus(self.sage2(xi, edge_index)).add(1e-3).clamp(max=1e2)
                hidden.append((loc, scale))
            loc, scale = zip(*hidden)
            loc = torch.stack(loc, dim=0)
            scale = torch.stack(scale, dim=0)
        else:
            loc = self.sage1(x, edge_index).clamp(min=-1e2, max=1e2)
            scale = F.softplus(self.sage2(x, edge_index)).add(1e-3).clamp(max=1e2)
        return loc, scale
    
class DiagBernoulliNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = DiagBernoulliNet([3, 4, 5])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=0, max=11)
        return logits

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class PreWhitener(nn.Module):
    """
    Data pre-whitener.
    """
    def __init__(self, data):
        super().__init__()
        with torch.no_grad():
            loc = data.mean(0)
            scale = data.std(0)
            scale[~(scale > 0)] = 1.
            self.register_buffer("loc", loc)
            self.register_buffer("inv_scale", scale.reciprocal())

    def forward(self, data):
        return (data - self.loc) * self.inv_scale
        
class DiagStudentTNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``df,loc,scale``
    triple, with shared ``df > 1``.

    This is used to represent a conditional probability distribution of a
    single Student's t random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = StudentTNet([3, 4])
        x = torch.randn(3)
        df, loc, scale = net(x)
        y = net.make_dist(df, loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes + [self.dim * 2])
        self.df_unconstrained = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)
        return df, loc, scale

    @staticmethod
    def make_dist(df, loc, scale):
        return dist.StudentT(df, loc, scale)


class DiagLaplaceNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Laplace random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = LaplaceNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.dim = sizes[-1]
        self.fc = FullyConnected(sizes + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Laplace(loc, scale)


class DiagExponentialNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``rate``.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = ExponentialNet([3, 4])
        x = torch.randn(3)
        rate, = net(x)
        y = net.make_dist(rate).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.dim = sizes[-1]
        self.fc = FullyConnected(sizes + [self.dim])

    def forward(self, x):
        scale = nn.functional.softplus(self.fc(x).squeeze(-1)).clamp(min=1e-3, max=1e6)
        rate = scale.reciprocal()
        return rate

    @staticmethod
    def make_dist(rate):
        return dist.Exponential(rate)


class DiagGammaNet(DistributionNet):

    def __init__(self, sizes):
        assert len(sizes) >= 1
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes + [self.dim * 2])

    def forward(self, x):
        concrate = nn.functional.softplus(self.fc(x))
        conc = nn.functional.softplus(concrate[..., :self.dim]).clamp(min=-1e6, max=1e6)
        rate = nn.functional.softplus(concrate[..., self.dim:]).clamp(min=1e-3, max=1e6).reciprocal()
        return conc, rate

    @staticmethod
    def make_dist(conc, rate):
        return dist.Gamma(conc, rate)


