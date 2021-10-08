import torch
from torch import nn
from torch.nn import functional as F


class Loss(nn.Module):
    def __init__(self, full_weight, grad_weight, occ_prob_weight):
        super().__init__()
        self.full_weight = full_weight
        self.grad_weight = grad_weight
        self.occ_prob_weight = occ_prob_weight
        self.l1_loss = nn.L1Loss(reduction='sum')
    
    def get_rgb_full_loss(self,rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1])
        return rgb_loss

    def get_smooth_loss(self, diff_norm):
        if diff_norm is None or diff_norm.shape[0]==0:
            return torch.tensor(0.0).cuda().float()
        else:
            return diff_norm.mean()

    def forward(self, rgb_pred, rgb_gt, diff_norm):
        rgb_gt = rgb_gt.cuda()
        
        if self.full_weight != 0.0:
            rgb_full_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt)
        else:
            rgb_full_loss = torch.tensor(0.0).cuda().float()

        if diff_norm is not None and self.grad_weight != 0.0:
            grad_loss = self.get_smooth_loss(diff_norm)
        else:
            grad_loss = torch.tensor(0.0).cuda().float()

        loss = self.full_weight * rgb_full_loss + \
               self.grad_weight * grad_loss
        if torch.isnan(loss):
            breakpoint()

        return {
            'loss': loss,
            'fullrgb_loss': rgb_full_loss,
            'grad_loss': grad_loss,
        }


