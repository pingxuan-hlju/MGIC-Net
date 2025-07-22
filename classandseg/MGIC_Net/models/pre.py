import torch
from torch import nn

class Pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(5,32),
                            nn.ReLU(),
                            nn.Linear(32,2))

    def forward(self,small_pre,mid_pre,count):
        dt = torch.cat((small_pre,mid_pre,count),dim=-1).to("cuda")
        output = self.out(dt)
        return output