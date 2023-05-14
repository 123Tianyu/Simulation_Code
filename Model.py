import torch.nn as nn
class DNN_Network(nn.Module):
    def __init__(self,input_size,output_size):
        super(DNN_Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

