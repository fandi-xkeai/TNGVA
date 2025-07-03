'''
***Closely based on code for TVAE by Matthew J. Vowels***
https://github.com/matthewvowels1/TVAE_release
See also the paper:
Vowels, M. J., Camgoz, N. C., & Bowden, R. (2021). Targeted VAE: Variational 
and targeted learning for causal inference. https://arxiv.org/pdf/2009.13472
'''
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan
from networks import GraphSAGENet,DiagNormalNet, DistributionNet, BernoulliNet, DiagBernoulliNet, FullyConnected,DiagGraphSAGENet, GraphSAGENet_residual,GraphSAGENet_BernoulliNet_residual,DiagGraphSAGENet_residual
from pygcn.layers import GraphConvolution


class Guide(PyroModule):
    def __init__(self, config):
        self.latent_dim_o = config["latent_dim_o"]
        self.latent_dim_c = config["latent_dim_c"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]

        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        super().__init__()
        self.t_nn = BernoulliNet([config["latent_dim_c"] + config["latent_dim_t"]])
        self.y_nn = FullyConnected([config["latent_dim_c"] + config["latent_dim_y"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        self.y0_nn = OutcomeNet([config["hidden_dim"]])
        self.y1_nn = OutcomeNet([config["hidden_dim"]])

        self.zc_nn = GraphSAGENet([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU()).cuda()
        self.zc_out_nn = DiagGraphSAGENet(config["hidden_dim"], config["latent_dim_c"]).cuda()
        self.zt_nn = GraphSAGENet([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU()).cuda()
        self.zt_out_nn = DiagGraphSAGENet(config["hidden_dim"], config["latent_dim_t"]).cuda()
        self.zy_nn = GraphSAGENet([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU()).cuda()
        self.zy_out_nn = DiagGraphSAGENet(config["hidden_dim"], config["latent_dim_y"]).cuda()
        self.zo_nn = GraphSAGENet([config["feature_dim"]] +
                                    [config["hidden_dim"]] * (config["num_layers"] - 1),
                                    final_activation=nn.ELU()).cuda()
        self.zo_out_nn = DiagGraphSAGENet(config["hidden_dim"], config["latent_dim_o"]).cuda()
        
        self.num_layers = config["num_layers"]
        
    def forward(self, x, adj, t=None, y=None, size=None):
        if adj is not None:
            if size is None:
                size = x.size(0)
            with pyro.plate("data", size, subsample=x):
                zo = pyro.sample("zo", self.zo_dist(x, adj)[1])
                zc = pyro.sample("zc", self.zc_dist(x, adj)[1])
                zt = pyro.sample("zt", self.zt_dist(x, adj)[1])
                zy = pyro.sample("zy", self.zy_dist(x, adj)[1])
                t = pyro.sample("t", self.t_dist(zc, zt), obs=t, infer={"is_auxiliary": True})
                y = pyro.sample("y", self.y_dist(t, zc, zy), obs=y, infer={"is_auxiliary": True})
            
    def z_mean(self, x, adj, t=None):
        with pyro.plate("data", x.size(0)):
            zo = pyro.sample("zo", self.zo_dist(x, adj)[1])
            zc = pyro.sample("zc", self.zc_dist(x, adj)[1])
            zt = pyro.sample("zt", self.zt_dist(x, adj)[1])
            zy = pyro.sample("zy", self.zy_dist(x, adj)[1])
        return zo,zc,zt,zy
    
    def z_loc(self, x, adj, t=None):
        return self.zo_dist(x, adj)[0], self.zc_dist(x, adj)[0], self.zt_dist(x, adj)[0], self.zy_dist(x, adj)[0]
    
    def t_dist(self, zc, zt):
        input_concat = torch.cat((zc, zt), -1)
        logits, = self.t_nn(input_concat)
        return dist.Bernoulli(logits=logits)

    def y_dist(self, t, zc, zy):
        x = torch.cat((zc, zy), -1)
        hidden = self.y_nn(x)
        params0 = self.y0_nn(hidden)
        params1 = self.y1_nn(hidden)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def zc_dist(self, x, adj):
        hidden = self.zc_nn(x.float(), adj)  
        params = self.zc_out_nn(hidden, adj)
        return params[0], dist.Normal(*params).to_event(1)

    def zt_dist(self, x, adj):
        hidden = self.zt_nn(x.float(), adj)
        params = self.zt_out_nn(hidden, adj)
        return params[0], dist.Normal(*params).to_event(1)

    def zy_dist(self, x, adj):
        hidden = self.zy_nn(x.float(), adj)
        params = self.zy_out_nn(hidden, adj)
        return params[0], dist.Normal(*params).to_event(1)

    def zo_dist(self, x, adj):
        hidden = self.zo_nn(x.float(), adj)
        params = self.zo_out_nn(hidden, adj)
        return params[0], dist.Normal(*params).to_event(1)



class Model(PyroModule):
    def __init__(self, config):
        self.latent_dim_o = config["latent_dim_o"]
        self.latent_dim_c = config["latent_dim_c"]
        self.latent_dim_t = config["latent_dim_t"]
        self.latent_dim_y = config["latent_dim_y"]
        self.contfeats = config["continuous_dim"]

        super().__init__()
        self.x_nn = GraphSAGENet_residual([config["latent_dim_c"] + config["latent_dim_t"] + config["latent_dim_y"] + config["latent_dim_o"]] +
                                    [config["hidden_dim"]] * (config["num_layers"]-1),
                                    final_activation=nn.ELU()).cuda()
        self.x_out_nn = DiagGraphSAGENet_residual(config["hidden_dim"], len(config["continuous_dim"])).cuda()

        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        self.y0_nn = OutcomeNet([config["latent_dim_c"] + config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.y1_nn = OutcomeNet([config["latent_dim_c"] + config["latent_dim_y"]] +
                                [config["hidden_dim"]] * config["num_layers"])
        self.t_nn = BernoulliNet([config["latent_dim_c"] + config["latent_dim_t"]])
        
    def forward(self, x, adj, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            zo = pyro.sample("zo", self.zo_dist())
            zc = pyro.sample("zc", self.zc_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
            x_continuous = pyro.sample("x_cont", self.x_dist_continuous(zo, zc, zt, zy, adj), obs=x[:, self.contfeats])
            t = pyro.sample("t", self.t_dist(zc, zt), obs=t)
            y = pyro.sample("y", self.y_dist(t, zc, zy), obs=y)
        return y

    def zo_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_o]).to_event(1)

    def zc_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_c]).to_event(1)

    def zt_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_t]).to_event(1)

    def zy_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim_y]).to_event(1)

    def x_dist_continuous(self, zo, zc, zt, zy, adj):
        z_concat = torch.cat((zo, zc, zt, zy), -1)  
        hidden = self.x_nn(z_concat, adj) 
        loc, scale = self.x_out_nn(hidden, adj)  
        return dist.Normal(loc, scale).to_event(1) 

    def x_dist_binary(self, zo, zc, zt, zy):
        z_concat = torch.cat((zo, zc, zt, zy), -1)
        logits = self.x2_nn(z_concat)
        return dist.Bernoulli(logits=logits).to_event(1)

    def y_dist(self, t, zc, zy):
        z_concat = torch.cat((zc, zy), -1)
        params0 = self.y0_nn(z_concat)
        params1 = self.y1_nn(z_concat)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def t_dist(self, zc, zt):
        z_concat = torch.cat((zc, zt), -1)
        logits, = self.t_nn(z_concat)
        return dist.Bernoulli(logits=logits)

    def y_mean(self, x, adj,t=None):
        with pyro.plate("data", x.size(0)):
            zo = pyro.sample("zo", self.zo_dist())
            zc = pyro.sample("zc", self.zc_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
            x_continuous = pyro.sample("x_cont", self.x_dist_continuous(zo, zc, zt, zy, adj), obs=x[:, self.contfeats])
            t = pyro.sample("t", self.t_dist(zc, zt), obs=t)
        return self.y_dist(t, zc, zy).mean

    def t_mean(self, x, adj):
        with pyro.plate("data", x.size(0)):
            zo = pyro.sample("zo", self.zo_dist())
            zc = pyro.sample("zc", self.zc_dist())
            zt = pyro.sample("zt", self.zt_dist())
            zy = pyro.sample("zy", self.zy_dist())
            x_continuous = pyro.sample("x_cont", self.x_dist_continuous(zo, zc, zt, zy, adj), obs=x[:, self.contfeats])
        return self.t_dist(zc, zt).mean
