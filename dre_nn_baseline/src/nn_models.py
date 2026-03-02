import torch
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden=64, depth=4, activation="tanh"):
        super().__init__()
        act = get_activation(activation)
        layers = [nn.Linear(in_dim, hidden), act]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        # t: [N,1]
        return self.net(t)