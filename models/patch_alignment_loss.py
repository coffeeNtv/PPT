import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAlignmentLoss(nn.Module):
    """
    Compute the patch alignment loss between two images inputs A and targets B

    Args:
        A: numpy array representing image A of size (size, size)
        B: numpy array representing image B of size (size, size)
        height: int, height of the rectangular region
        width: int, width of the rectangular region

    Returns:
        float, the computed loss between A and B
    """
    def __init__(self, height=4, width=4):
        super(PatchAlignmentLoss, self).__init__()
        self.height = height
        self.width = width
        self.beta = 0.0025

    def forward(self, A, B):
        assert A.size() == B.size(), "A and B must have the same size"
        # Compute the distances between patch A and B
        A_patches = F.unfold(A, kernel_size=(self.height, self.width), padding=(self.height // 2, self.width // 2), stride=self.height)
        B_patches = F.unfold(B, kernel_size=(self.height, self.width), padding=(self.height // 2, self.width // 2), stride=self.height)
        distances = torch.sum(torch.abs(B_patches - A_patches), dim=1) 
        loss = torch.mean(distances)*self.beta
        return loss
