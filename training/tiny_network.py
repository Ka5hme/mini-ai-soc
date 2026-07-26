import torch.nn as nn

class tinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(8, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x
