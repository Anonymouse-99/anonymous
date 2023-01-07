import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import math

class GCN(nn.Module):
    def __init__(self, in_features, out_features, num_node=None, bias=True, input_vector=False):
        super(GCN, self).__init__()
        # Initialize parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.input_vector = input_vector
        self.num_node = num_node
        
        # Initialize weights
        self.weight = Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()
        
        # Edge index mapping from vector to matrix
        if input_vector:
            t = 0
            self.pair_i = []
            self.pair_j = []
            self.pair_t = []
            for i in range(self.num_node):
                for j in range(i+1, self.num_node):
                    self.pair_i.append(i)
                    self.pair_j.append(j)
                    self.pair_t.append(t)
                    t += 1


    def _reset_parameters(self):
        if (self.in_features == self.out_features):
            init.orthogonal_(self.weight)
        else:
            init.uniform_(self.weight,
                          a=-math.sqrt(1.0 / self.in_features) * math.sqrt(3),
                          b=math.sqrt(1.0 / self.in_features) * math.sqrt(3))
        if self.bias is not None:
            init.uniform_(self.bias, -0, 0)


    def forward(self, X, A):
        '''
        A: [N,V,V] or [V,V]
        X: [N,V,F]
        '''
        if self.input_vector:
            A = self.vector2matrix(A)
            return torch.matmul(torch.matmul(A, X), self.weight) + self.bias, A
        else:
            return torch.matmul(torch.matmul(A, X), self.weight) + self.bias


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
    

    def vector2matrix(self, A):
        '''
        Mapping from vector to 2D matrix
        A: [N, num_edges] or [num_edges]
        '''
        if len(A.shape)==2:
            # [N, num_edges]
            A_M = torch.zeros(A.shape[0], self.num_node, self.num_node).cuda()
            A_M[:, self.pair_i, self.pair_j] = A[:, self.pair_t]
            A_M[:, self.pair_j, self.pair_i] = A[:, self.pair_t]
        else:
            # [num_edges]
            A_M = torch.zeros(self.num_node, self.num_node).cuda()
            A_M[self.pair_i, self.pair_j] = A[self.pair_t]
            A_M[self.pair_j, self.pair_i] = A[self.pair_t]

        # Add self-connection of nodes
        A_M += torch.eye(self.num_node).cuda()
        return A_M
