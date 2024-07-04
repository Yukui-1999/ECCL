import torch
import torch.nn as nn
from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)

class tabMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.act = act_layer()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.drop2 = nn.Dropout(drop_probs[0])
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop3 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.drop3(x)
        return x
    
# from torchsummary import summary
# model = tabMlp(195, 512,512,drop=0.1)
# summary(model, (195,),device='cpu')