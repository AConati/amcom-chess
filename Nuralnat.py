import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import chess
from move_mask import move_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 80
batch_size = 2048
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

    def forward(self, data):
        residual = data
        out = self.conv1(data)
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

        # policy head
        # possibly bad, maybe more lairs?
        self.conv_p1 = nn.Conv2d(self.in_channels, filters, kernel_size=1)
        self.bn_p1 = nn.BatchNorm2d(filters)
        self.relu_p1 = nn.ReLU(inplace=True)

        self.conv_p2 = nn.Conv2d(self.in_channels, 73, kernel_size=1)
        self.bn_p2 = nn.BatchNorm2d(73)
        #Arbitrarily define what layers mean
        self.relu_p2 = nn.ReLU(inplace=True)
        self.flatten_p2 = nn.Flatten()

        # value head
        self.conv_v = nn.Conv2d(self.in_channels, 1, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(1)
        self.relu_v = nn.ReLU(inplace=True)
        self.flatten_v = nn.Flatten()
        self.layer_v = nn.linear(64, 32)
        self.relu2_v = nn.ReLU(inplace=True)
        self.layer2_v = nn.linear(32, 1)
        self.tanh_v = nn.Tanh()
        

    def make_layer(self, residLayer, out_channels, layers, stride=1):
        layers = []
        for x in range(0, layers):
            layers.append(residLayer(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data):
        out = self.conv(data)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer(out)

        pout = self.conv_p1(out)
        pout = self.bn_p1(pout)
        pout = self.relu_p1(pout)
        pout = self.conv_p2(pout)
        pout = self.bn_p2(pout)
        pout = self.relu_p2(pout)
        pout = self.flatten_p2(pout)
        legal_moves, move_values = move_mask(pout)
        pout = F.softmax(move_values)

        vout = self.conv_v(out)
        vout = self.bn_v(vout)
        vout = self.relu_v(vout)
        vout = self.flatten_v(vout)
        vout = self.layer_v(vout)
        vout = self.relu2_v(vout)
        vout = self.layer2_v(vout)
        vout = self.tanh_v(vout)

        return legal_moves, pout, vout


#Create model on GPU and pass to train
model = AmnomZero(ResidualLayer, 10, filters = 128).to(device)

#loss function(s)

optimize = torch.optim.AdamW(model.parameters(), lr = learning_rate)


