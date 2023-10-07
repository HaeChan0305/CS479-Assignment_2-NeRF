"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()
        
        # TODO
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(pos_dim, feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.fc4 = nn.Linear(feat_dim, feat_dim)
        self.fc5 = nn.Linear(feat_dim, feat_dim)
        self.fc6 = nn.Linear(feat_dim + pos_dim, feat_dim)
        self.fc7 = nn.Linear(feat_dim, feat_dim)
        self.fc8 = nn.Linear(feat_dim, feat_dim)
        self.fc9 = nn.Linear(feat_dim, feat_dim)
        self.fc_additional = nn.Linear(feat_dim, 1)
        self.fc10 = nn.Linear(feat_dim + view_dir_dim, feat_dim//2)
        self.fc11 = nn.Linear(feat_dim//2, 3)

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        x = self.fc1(pos)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        x = self.relu(x)
        
        x = self.fc5(x)
        x = self.relu(x)
        
        x = torch.cat((x, pos), -1)
        x = self.fc6(x)
        x = self.relu(x)
        
        x = self.fc7(x)
        x = self.relu(x)
        
        x = self.fc8(x)
        x = self.relu(x)
        
        x = self.fc9(x)
        sigma = self.fc_additional(x)
        sigma = self.relu(sigma)
        x = torch.cat((x, view_dir), -1)
        
        x = self.fc10(x)
        x = self.relu(x)
        
        x = self.fc11(x)
        radiance = self.sigmoid(x)
    
        return (sigma, radiance)
