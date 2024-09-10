"""
From https://github.com/vlukiyanov/pt-dec
"""

import torch
import torch.nn as nn
from typing import Tuple
from .cluster import ClusterAssignment

class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        orthogonal=True,
        freeze_center=True, project_assignment=True
    ):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment
        )

        self.loss_fn = nn.KLDivLoss(size_average=False)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_num = batch.size(1)
        batch_size = batch.size(0)
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        encoded = encoded.view(batch_size * node_num, -1)
        assignment = self.assignment(encoded)
        assignment = assignment.view(batch_size, node_num, -1)
        encoded = encoded.view(batch_size, node_num, -1)
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def loss(self, assignment):
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.log(), target) / flattened_assignment.size(0)

    def get_cluster_centers(self) -> torch.Tensor:
        
        return self.assignment.get_cluster_centers()
