from __future__ import absolute_import, print_function

# --- System ---
import os
import sys
import warnings

# --- Utility ---
import pandas as pd
import numpy as np
import math
import random
import logging
import pickle
from sklearn.model_selection import train_test_split

# --- Plot --
import matplotlib.pyplot as plt
import seaborn as sns

# --- Pytorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import random_split 

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
root_dir = os.getcwd()

dataPath = "data/statistics (pacing).csv"
df = pd.read_csv(dataPath)
# columnList = df.columns

# Dropping columns that are not required at the moment
df = df.drop(columns=[ 'Unnamed: 0', 'UUID', 'HOSTNAME', 'ALIAS', 'TIMESTAMP',
                       'THROUGHPUT (Receiver)', 'LATENCY (min.)', 'LATENCY (max.)', 
                       'CONGESTION (Receiver)', 'BYTES (Receiver)'
                     ])

# Pre-processing
pacing = df['PACING'].values
for i, p in enumerate(pacing):
    v, _ = p.split("gbit")
    pacing[i] = int(v)

df['PACING'] = pacing
df['CONGESTION (Sender)'] = (df['CONGESTION (Sender)'] == 'cubic').astype(int)

X = df[['THROUGHPUT (Sender)', 'LATENCY (mean)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)']].values
y = df['PACING'].values
y = y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

EPOCH = 55
BATCH = 2
LEARNING_RATE = 0.000012

traindata   = TensorDataset(torch.Tensor(X_train),
                            torch.Tensor(y_train)
                           )
trainloader = DataLoader(traindata,
                        batch_size = BATCH,
                        shuffle = True
                        )

testdata   = TensorDataset(torch.Tensor(X_test),
                           torch.Tensor(y_test)
                          )
testloader = DataLoader(testdata,
                        batch_size = BATCH,
                        shuffle = True
                       )

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear (5, 32)
        self.fc2 = torch.nn.Linear (32, 32)
        self.fc3 = torch.nn.Linear (32, 1)
        self.sig = torch.nn.Sigmoid()
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x

model = Network()
# print( f"====================\nTotal params: {len(list(model.parameters()))}\n====================" )

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
bestloss = 10

def train(epoch):

    acc, correct, loss = 0.0, 0.0, 0.0
    running_loss, total = 0.0, 0

    model.train()
    
    for xs, ys in trainloader:
        xs, ys = xs.to(device), ys.to(device)
        
        # --- Model ---
        optimizer.zero_grad()
        output =  model(xs)
        
        # --- Loss ---
        loss = criterion(ys, output)
        loss.backward()
        optimizer.step()
        
        # --- Statistics ---
        running_loss += loss.item() * xs.size(0)

        _, predicted = torch.max(output.data, 1)
        total += ys.size(0)
        correct += (predicted == ys).sum().item()
    
    epoch_loss  = running_loss/len(traindata)
    acc = (100 * correct / total)
    return epoch_loss, acc

def test(epoch):
    
    acc, correct, loss = 0.0, 0.0, 0.0
    running_loss, total = 0.0, 0
    
    model.eval()
    
    with torch.no_grad():
        for xs, ys in testloader:
            xs, ys = xs.to(device), ys.to(device)
            
            # --- Model ---
            output = model(xs)
            
            # --- Loss ---
            loss = criterion(ys, output)
            
            # --- Statistics ---
            running_loss += loss.item() * xs.size(0)
            
            _, predicted = torch.max(output.data, 1)
            total += ys.size(0)
            correct += (predicted == ys).sum().item()
            print(predicted, ys)

        epoch_loss  = running_loss/len(testdata)
    acc = (100 * correct / total)
    return epoch_loss, acc
    
if not os.path.isdir(str(root_dir)+'/checkpoint'):
    os.mkdir(str(root_dir)+'/checkpoint')

print()
print("Epoch", "TR-loss", "TS-loss", sep=' '*8, end="\n")

for epoch in range(EPOCH):
    
    trainloss, trainacc = train(epoch)
    testloss, testacc  = test(epoch)

    print(f"{epoch+0:03}/{EPOCH}", f"{trainloss:.4f}", f"{testloss:.4f}", f"{trainacc:.4f}", f"{testacc:.4f}", sep=' '*8, end="\n")
    
    # # Saving the model.
    # is_best = testloss < bestloss
    # bestloss = min(testloss, bestloss)
    # if is_best:
    #     torch.save(model.state_dict(), str(root_dir)+"/checkpoint/pacing_"+str(epoch)+".pt")
    #     print("Model Saved.")
print("="*50)