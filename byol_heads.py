import torch
from torch import nn

class ProjectionHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(7296, 30)
        self.lin1 = nn.Linear(19712, 100)
        self.lin2 = nn.Linear(11904, 40)
        self.lin3 = nn.Linear(7296, 30)
    
    def forward(self, x0, x1, x2, x3):
        x0 = self.lin0(x0)
        x1 = self.lin1(x1)
        x2 = self.lin2(x2)
        x3 = self.lin3(x3)

        return torch.cat([x0, x1, x2, x3], dim=1)


class PredictionHead(nn.Sequential):

    def __init__(self):
        super().__init__([
            nn.Linear(200, 20),
            nn.ReLU(),
            nn.Linear(20, 200),
        ])