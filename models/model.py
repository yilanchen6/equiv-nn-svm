import random
import numpy as np
import torch
from torch import nn


def _initialize_weights(model, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    for i, m in enumerate(model):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 1.0)


def make_layers(hidden_layer=1, width=512, input_size=3 * 32 * 32, output_size=10):
    layers = [nn.Linear(input_size, width, bias=False), nn.ReLU()]

    if hidden_layer >= 1:
        for i in range(hidden_layer):
            layers += [nn.Linear(width, width, bias=False), nn.ReLU()]

    layers += [nn.Linear(width, output_size, bias=False)]
    return nn.Sequential(*layers)


class FCNN(nn.Module):
    def __init__(self, width=400, hidden_layer=1, seed=0):
        super(FCNN, self).__init__()
        self.scale = width ** 0.5
        self.classifier = make_layers(hidden_layer=hidden_layer, width=width, input_size=28 * 28, output_size=1)
        _initialize_weights(self.classifier, seed=seed)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i, m in enumerate(self.classifier):
            if isinstance(m, nn.Linear):
                if i == 0:
                    x = m(x) / (x.size()[-1] ** 0.5)
                else:
                    x = m(x) / self.scale
            else:
                x = m(x)
        return x
