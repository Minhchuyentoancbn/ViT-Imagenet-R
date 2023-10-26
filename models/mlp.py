import torch
import torch.nn as nn

from .base_model import BaseModel


class MLP(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        layers = []
        for i in range(len(args.hidden_layer)):
            if i == 0:
                layers.append(nn.Linear(768, args.hidden_layer[i]))
            else:
                layers.append(nn.Linear(args.hidden_layer[i-1], args.hidden_layer[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(args.hidden_layer[-1], 200))
        if args.dropout > 0:
            layers.append(nn.Dropout(args.dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x