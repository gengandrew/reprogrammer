import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(input: torch.Tensor, target: torch.Tensor, reduction: str):
    loss = -F.cosine_similarity(input, target)

    if reduction == 'none':
        loss = loss
    elif reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha*cosine_similarity(input, target)