# https://github.com/ShannonAI/dice_loss_for_NLP/blob/master/loss/focal_loss.py


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """
    Focal loss(https://arxiv.org/pdf/1708.02002.pdf)
    Shape:
        - input: (N, C)
        - target: (N)
        - Output: Scalar loss
    Examples:
        >>> loss = FocalLoss(gamma=2, alpha=[1.0]*7)
        >>> input = torch.randn(3, 7, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(7)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, gamma=0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # [N, 1]
        target = target.unsqueeze(-1)
        # [N, C]
        pt = F.softmax(input, dim=-1)
        logpt = F.log_softmax(input, dim=-1)
        # [N]
        pt = pt.gather(1, target).squeeze(-1)
        logpt = logpt.gather(1, target).squeeze(-1)

        if self.alpha is not None:
            # [N] at[i] = alpha[target[i]]
            at = self.alpha.gather(0, target.squeeze(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "none":
            return loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

    @staticmethod
    def convert_binary_pred_to_two_dimension(x, is_logits=True):
        """
        Args:
            x: (*): (log) prob of some instance has label 1
            is_logits: if True, x represents log prob; otherwhise presents prob
        Returns:
            y: (*, 2), where y[*, 1] == log prob of some instance has label 0,
                             y[*, 0] = log prob of some instance has label 1
        """
        probs = torch.sigmoid(x) if is_logits else x
        probs = probs.unsqueeze(-1)
        probs = torch.cat([1-probs, probs], dim=-1)
        logprob = torch.log(probs+1e-4)  # 1e-4 to prevent being rounded to 0 in fp16
        return logprob

    def __str__(self):
        return f"Focal Loss: gamma={self.gamma}, alpha={self.alpha}"

    def __repr__(self):
        return str(self)
    
# output_gamma2_alpha = FocalLoss(gamma=2, alpha=torch.tensor([0.7, 0.3]))(input, target)
# output_gamma0_alpha = FocalLoss(gamma=0, alpha=torch.tensor([0.7, 0.3]))(input, target)
# output_gamma0_alpha1 = FocalLoss(gamma=1, alpha=torch.tensor([1.0]*2))(input, target)
# output_gamma0_alpha05 = FocalLoss(gamma=1, alpha=torch.tensor([0.5]*2))(input, target)
# output_gamma2_alpha, output_gamma0_alpha, output_gamma0_alpha1, output_gamma0_alpha05