import torch
from torch import nn

class Assign_weight(nn.Module):
    def __init__(self):
        super(Assign_weight, self).__init__()
        self.seg_emb = nn.Linear(1,64)
        self.lay_emb = nn.Linear(1,64)
        self.out = nn.Linear(128,1)

    def forward(self,segout_shape,small_layer,mid_layer,large_layer):
        B = segout_shape.shape[0]
        small_layer = torch.tensor(small_layer).view(1, 1).expand(B, 1).to("cuda")
        mid_layer = torch.tensor(mid_layer).view(1, 1).expand(B, 1).to("cuda")
        large_layer = torch.tensor(large_layer).view(1, 1).expand(B, 1).to("cuda")
        seg_emb = self.seg_emb(segout_shape)
        small_layer = self.lay_emb(small_layer)
        mid_layer = self.lay_emb(mid_layer)
        large_layer = self.lay_emb(large_layer)
        small_weight = self.out(torch.cat([seg_emb,small_layer],dim=1))
        mid_weight = self.out(torch.cat([seg_emb,mid_layer],dim=1))
        large_weight = self.out(torch.cat([seg_emb,large_layer],dim=1))
        return torch.cat([small_weight,mid_weight,large_weight],dim=1)