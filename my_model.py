
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        #self.conv_layer = nn.Sequential(
        self.conv1= nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5= nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6= nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.drop1 = nn.Dropout2d(p=0.01)
        self.linear1 = nn.Linear(256 *4 *4, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(p=0.01)
        self.linear3 = nn.Linear(512, 11)
        #)
        

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        # x = self.pool(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = self.drop1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        # x = self.pool(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.pool(x)
        #x = self.drop1(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)

        #print(x.shape)
        # x = x.view(-1, 256 * 2 * 2)
        x = torch.flatten(x, 1)
        x = self.drop2(x)
        # print(x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        #x = self.drop2(x)
        x = self.linear3(x)
        outs = x
        
        return outs       