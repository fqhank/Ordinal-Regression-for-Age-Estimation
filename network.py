import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class AgeNet(nn.Module):
    def __init__(self,min_age=15,max_age=72):
        super(AgeNet,self).__init__()
        self.min_age = min_age
        self.max_age = max_age
        self.feature_net = nn.Sequential(
            nn.Conv2d(3,20,kernel_size=5,stride=1,padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(20,40,kernel_size=7,stride=1,padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(40,80,kernel_size=11,stride=1,padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(80),
            nn.Flatten(),
            nn.Linear(80,80),
            # nn.Dropout(0.5),
        )
        self.fc_layers = []
        for i in range(min_age,max_age):
            exec('self.FC2_{}=nn.Linear(80,2)'.format(i))
            exec('self.fc_layers.append(self.FC2_{})'.format(i))
        self.softmax = nn.Softmax(dim=2)
        # self._init_parameters()

    def forward(self,input):
        feature = self.feature_net(input)
        out = self.softmax(self.fc_layers[0](feature).unsqueeze(1))
        for i in range(self.min_age+1,self.max_age):
            temp = self.softmax(self.fc_layers[i-self.min_age](feature).unsqueeze(1))
            out = torch.cat((out,temp),1)
        return out

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

