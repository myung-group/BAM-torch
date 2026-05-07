"""
Defines neural network layers in BPNN (PyTorch version)

Each class represents a fully connected layer
"""

import torch
import torch.nn as nn


class ReLU(nn.Module):
    def __init__(self, n_in, n_out, parameters=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        if parameters is None:
            self.W = nn.Parameter(torch.empty(n_in, n_out))
            self.b = nn.Parameter(torch.empty(1, n_out))

            # Xavier initializer (TF contrib.xavier_initializer)
            nn.init.xavier_uniform_(self.W)
            nn.init.xavier_uniform_(self.b)
        else:
            self.W = parameters["W"]
            self.b = parameters["b"]

    def __repr__(self):
        return f"ReLU({self.n_in}, {self.n_out})"

    def forward(self, nn_input):
        return torch.relu(nn_input @ self.W + self.b)
