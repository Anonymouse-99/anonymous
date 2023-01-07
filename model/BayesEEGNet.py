import torch
import torch.nn as nn

from .FeatureNet import FeatureNet_MC
from .GraphBuilder import GraphBuilder


class BayesEEGNet(nn.Module):
    def __init__(self, hidden_size, output_size,
                 graph_node_dim, num_nodes, last_dense=64, res=True):
        super(BayesEEGNet, self).__init__()
        self.hidden_size = hidden_size
        self.graph_node_dim = graph_node_dim
        self.num_nodes = num_nodes
        self.res = res
        
        self.feature = FeatureNet_MC(s_freq=100, filters=128, dropout=0.5)

        if self.hidden_size!=512:
            self.changeDim = nn.Linear(512, self.hidden_size)

        self.graph = GraphBuilder(hidden_size = self.hidden_size,
                                  graph_node_dim = self.graph_node_dim,
                                  num_nodes = num_nodes,
                                  dropout=0.1,
                                  )

        self.classifier = nn.Sequential(
            nn.Linear(self.graph_node_dim * self.num_nodes, last_dense),
            nn.Dropout(p=0.1),
            nn.Linear(last_dense, output_size))

    def forward(self, x, predict_only=False):
        B, C, T = x.size()
        
        # Feature extract
        x = self.feature(x)
        if self.hidden_size!=512:
            x = self.changeDim(x)

        # Build graph representation
        graph_out = self.graph(x, predict_only)
        x = graph_out['H_g']
        
        # Classify
        x = torch.reshape(x, (B, -1))
        x = self.classifier(x)

        graph_out['y_hat'] = x
        return graph_out
