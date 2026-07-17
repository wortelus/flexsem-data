import torch
from torch import nn

class RelativeMSELoss(nn.Module):
    def __init__(self, eps_nm: float, scaler):
        super().__init__()
        self.eps_nm = eps_nm
        self.register_buffer("scaler_scale", torch.tensor(float(scaler.scale_[0])))
        self.register_buffer("scaler_min", torch.tensor(float(scaler.min_[0])))

    def inverse_scale(self, data):
        return (data - self.scaler_min) / self.scaler_scale

    def forward(self, pred, target):
        pred_nm = self.inverse_scale(pred)
        target_nm = self.inverse_scale(target)
        scale = target_nm.norm(dim=-1, keepdim=True) + self.eps_nm
        return (((pred_nm - target_nm) / scale) ** 2).mean()