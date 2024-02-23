# based on:
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int, device: torch.device = None, dtype: torch.dtype = None, eps: float = 1e-7) -> torch.Tensor:
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float, reduction: str = 'none', eps: float = 1e-7) -> torch.Tensor:
    # Compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # Create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], eps=eps, device=input.device, dtype=input.dtype)

    # Compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 4.0, reduction: str = 'mean', eps: float = 1e-7) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)