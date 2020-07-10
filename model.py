import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution, GraphConvolutionSparse, Linear


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolutionSparse(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dc = InnerDecoder(dropout, act=lambda x: x)

        #for embedding attributes/features
        self.linear_a1= Linear(n_nodes, hidden_dim1, act = F.tanh,sparse_inputs=True) # the input dim is the number of nodes
        self.linear_a2= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)
        self.linear_a3= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)




    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        hidden_a1 = self.linear_a1(x.t()) # transpose the input feature matrix

        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), self.linear_a2(hidden_a1),self.linear_a3(hidden_a1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar, mu_a, logvar_a = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        z_a = self.reparameterize(mu_a,logvar_a)
        return self.dc((z,z_a)),mu, logvar, mu_a, logvar_a


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class InnerDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self,dropout=0., act=torch.sigmoid,**kwargs):
        super(InnerDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def forward(self,inputs):
        z_u, z_a = inputs
        z = F.dropout(z_u, self.dropout, training=self.training)
        adj = self.act(torch.mm(z_u, z_u.t()))
        features = self.act(torch.mm(z_u,z_a.t()))
        return adj,features
