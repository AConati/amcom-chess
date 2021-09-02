import treesearch
from Nuralnat import AmnomZero, ResidualLayer, CustomLoss
import nnclass
import move_mask
import train
import torch
import time

# Hyper parameters
num_epochs = 80
learning_rate = 0.001
c = .0001

t0 = time.time()
model = AmnomZero(ResidualLayer, 10, filters = 128).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=c, amsgrad=False)
criterion = CustomLoss()
t1 = time.time()

print(t1-t0)

g_list = train.make_trainSet(model, 1)