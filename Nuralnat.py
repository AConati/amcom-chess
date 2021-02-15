import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 80
batch_size = 100
learning_rate = 0.001

# Residual layer sub-model
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual #skip connection
        out = self.relu(out)
        return out

# Alpha Zero
# pass in
class AmnomZero(nn.Module):
    def __init__(self, residLayer, layers=40, filters=256, num_moves=73):
        super(AmnomZero, self).__init__()
        self.in_channels = 119  #https://arxiv.org/pdf/1712.01815.pdf page 13 hmm
        self.conv = nn.Conv2d(self.in_channels, filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.layer = self.make_layer(residLayer, filters, layers)
        #add value and policy head
        #add function for making residual layers

#Make our own loss function? also maybe not in here. Maybe this is just the nurlnat
#training loop is hard but maybe not in here

