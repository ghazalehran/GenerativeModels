from PIL import ImageFilter
import random

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Module

from torch import Tensor

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, stop_gradient=True, MLP_mode=None):
        """
        Args:
            base_encoder: Backbone encoder model.
            dim: Feature dimension (default: 2048).
            pred_dim: Hidden dimension of the predictor (default: 512).
            stop_gradient: Whether to stop gradient flow for targets.
            MLP_mode: Controls predictor behavior:
                - None: Default predictor behavior.
                - 'fixed_random_init': Predictor uses fixed random initialization.
                - 'no_pred_mlp': Predictor is replaced by an identity function.
        """
        super(SimSiam, self).__init__()

        self.stop_gradient = stop_gradient
        self.MLP_mode = MLP_mode

        # Create the encoder with a 3-layer projector
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False 

        # Create the predictor (2-layer MLP)
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer
        
        if self.MLP_mode=='fixed_random_init':
            # freeze all layers but the last fc
            for param in self.predictor.parameters():
                param.requires_grad = False
            # init the self.predictor layer
            self.predictor[0].weight.data.normal_(mean=0.0, std=0.01)
            self.predictor[3].weight.data.normal_(mean=0.0, std=0.01)
            self.predictor[3].bias.data.zero_()

        elif self.MLP_mode=='no_pred_mlp':
            self.predictor = nn.Identity()
        else:
            pass

    def forward(self, x1, x2):
        """
        Args:
            x1: First view of images.
            x2: Second view of images.
        Returns:
            p1, p2: Predictors for x1 and x2.
            z1, z2: Targets for x1 and x2.
        """
        # Encoder outputs
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        if self.stop_gradient:
        # stop gradient from backpropagating to the encoder
            z1 = z1.detach()
            z2 = z2.detach()

        # compute the predictors
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1, z2

    def loss (self, p1,p2,z1,z2, similarity_function='CosineSimilarity'):
        """
        Compute SimSiam loss.
        Args:
            p1, p2: Predictors (batch_size, dim).
            z1, z2: Targets (batch_size, dim).
            similarity_function: Similarity function to use. Default: 'CosineSimilarity'.
        Returns:
            SimSiam loss (scalar tensor).
        """
        with torch.no_grad():
            z2_stop = z2.clone()
            z1_stop = z1.clone()

        if similarity_function == 'CosineSimilarity':
            sim_func = CosineSimilarity(dim=1, eps=1e-8)
            loss = -sim_func(p1, z2.detach()).mean() / 2 - sim_func(p2, z1.detach()).mean() / 2
        else:
            raise ValueError(f"Invalid similarity function: {similarity_function}. Supported: ['CosineSimilarity']")
        return loss

# Utility function for batch-wise dot product (if needed)
def bdot(a, b):
    """
    Performs batch-wise dot product.
    Args:
        a, b: Tensors of shape (batch_size, dim).
    Returns:
        Batch-wise dot product as a tensor of shape (batch_size,).
    """
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

class CosineSimilarity(Module):
    """
    Computes cosine similarity between two tensors.
    Args:
        dim: Dimension along which to compute similarity (default: 1).
        eps: Small value to avoid division by zero (default: 1e-8).
    """
    __constants__ = ['dim', 'eps']

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute cosine similarity between x1 and x2.
        Args:
            x1, x2: Input tensors of shape (batch_size, dim).
        Returns:
            Tensor of cosine similarities.
        """
        return F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)