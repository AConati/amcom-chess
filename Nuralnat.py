import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import chess
from move_mask import move_mask
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 80
batch_size = 2048
learning_rate = 0.001
c = .0001

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
    def __init__(self, residLayer, board, layers=40, filters=256, num_moves=73):
        super(AmnomZero, self).__init__()
        self.board = board
        self.in_channels = 19  #https://arxiv.org/pdf/1712.01815.pdf page 13 hmm
        self.conv = nn.Conv2d(self.in_channels, filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.layer = self.make_layer(residLayer, filters, filters, layers)   # in channels has to match out channels of self.conv call which is filters here

        # policy head
        # possibly bad, maybe more lairs?
        self.conv_p1 = nn.Conv2d(filters, filters, kernel_size=3, padding = 1)
        self.bn_p1 = nn.BatchNorm2d(filters)
        self.relu_p1 = nn.ReLU(inplace=True)

        self.conv_p2 = nn.Conv2d(filters, 73, kernel_size=1)
        self.bn_p2 = nn.BatchNorm2d(73)


        #Arbitrarily define what layers mean
        self.relu_p2 = nn.ReLU(inplace=True)
        self.flatten_p2 = nn.Flatten()

        # value head
        self.conv_v = nn.Conv2d(filters,1, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(1)
        self.relu_v = nn.ReLU(inplace=True)
        self.flatten_v = nn.Flatten()
        self.layer_v = nn.Linear(64, 32)
        self.relu2_v = nn.ReLU(inplace=True)
        self.layer2_v = nn.Linear(32, 1)
        self.tanh_v = nn.Tanh()

    
        

    def make_layer(self, residLayer, in_channels, out_channels, numlayers, stride=1):
        layers = []
        for x in range(0, numlayers):
            layers.append(residLayer(in_channels, out_channels))

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
        pout = torch.squeeze(pout)
        legal_moves, move_values = move_mask(pout, self.board)
        move_values = torch.FloatTensor(move_values)
        pout = F.softmax(move_values, dim = 0)
        
        vout = self.conv_v(out)
        vout = self.bn_v(vout)
        vout = self.relu_v(vout)
        vout = self.flatten_v(vout)
        vout = self.layer_v(vout)
        vout = self.relu2_v(vout)
        vout = self.layer2_v(vout)
        vout = self.tanh_v(vout)

        #Do we care about legal moves as an output?
        return legal_moves, pout, vout


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mseLossFn = nn.MSELoss().to(device)
        

    def forward(self, model, endVal, neuralVal, mcProb, nnProb):
        # endval = z, neuralval = v, mcProb = pi, nnprob = p
        mseLoss = self.mseLossFn(endVal, neuralVal)
        test = torch.mul(nnProb, -torch.log(mcProb))
        ceLoss = torch.sum(torch.mul(nnProb, -torch.log(mcProb)))

        
        return mseLoss + ceLoss 




#Create model on GPU and pass to train
test_board = chess.Board()
model = AmnomZero(ResidualLayer, test_board, 10, filters = 128).to(device)
#loss function(s)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=c, amsgrad=False)
criterion = CustomLoss()
a = torch.rand([1,19,8,8]).to(device)
print(model)
a,b,c = model(a)
z = torch.rand([1,19,8,8]).to(device)
d,e,f = model(z)

loss = criterion(model,c,f,b,e)
print("TEST LOSS= " +str(loss.item()))
optimizer.zero_grad()
loss.backward()
optimizer.step()
#MODEL MAKES NUMBERS!
print(a)
print(b.numpy())
print(c.item())