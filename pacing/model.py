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
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# --- Plot --
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# --- Pytorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import random_split

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# Normalization
minmax_scale = preprocessing.MinMaxScaler().fit(df[['THROUGHPUT (Sender)', 'LATENCY (mean)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)']])
df_minmax = minmax_scale.transform(df[['THROUGHPUT (Sender)', 'LATENCY (mean)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)']])

final_df = pd.DataFrame(df_minmax, columns=['THROUGHPUT (Sender)', 'LATENCY (mean)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)'])
final_df.head(5)

X = final_df[['THROUGHPUT (Sender)', 'LATENCY (mean)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)']].values

EPOCH = 50
BATCH = 32
LEARNING_RATE = 0.001
SAVE = False
BESTLOSS = 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test  = torch.tensor(X_test)
y_test  = torch.tensor(y_test) 

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


# Dataset w/o any tranformations
traindata   = CustomTensorDataset(tensors=(X_train, y_train), transform=None)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=BATCH)

testdata    = CustomTensorDataset(tensors=(X_test, y_test), transform=None)
testloader = torch.utils.data.DataLoader(testdata, batch_size=BATCH)

class PacingOptimizer(nn.Module):
    def __init__(self):
        super(PacingOptimizer, self).__init__()
        self.fc1 = torch.nn.Linear (5, 32)
        self.fc2 = torch.nn.Linear (32, 64)
        self.fc3 = torch.nn.Linear (64, 32)
        self.fc4 = torch.nn.Linear (32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = PacingOptimizer()
# print( f"====================\nTotal params: {len(list(model.parameters()))}\n====================" )

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

def train(epoch):

    acc, correct, loss = 0.0, 0.0, 0.0
    running_loss, total = 0.0, 0

    model.train()

    for xs, ys in trainloader:
        xs, ys = xs.to(device).float(), ys.to(device).float()

        # --- Model ---
        optimizer.zero_grad()
        output =  model(xs)

        # --- Loss ---
        loss = criterion(ys, output)
        loss.backward()
        optimizer.step()

        # --- Statistics ---
        running_loss += loss.item() * xs.size(0)

        # _, predicted = torch.max(output.data, 1)
        # total += ys.size(0)
        # correct += (predicted == ys).sum().item()

    epoch_loss  = running_loss/len(traindata)
    # acc = (100 * correct / total)
    return epoch_loss #, acc

def test(epoch):

    acc, correct, loss = 0.0, 0.0, 0.0
    running_loss, total = 0.0, 0

    model.eval()

    with torch.no_grad():
        for xs, ys in testloader:
            xs, ys = xs.to(device).float(), ys.to(device).float()

            # --- Model ---
            output = model(xs)

            # --- Loss ---
            loss = criterion(ys, output)

            # --- Statistics ---
            running_loss += loss.item() * xs.size(0)

            # predicted = torch.max(output.data, 1)[1]
            # total += ys.size(0)
            # correct += (predicted == ys).sum().item()
            print(f"Pred: {output}, Target: {ys}")

        epoch_loss  = running_loss/len(testdata)
    # acc = (100 * correct / total)
    return epoch_loss #, acc

if not os.path.isdir(str(root_dir)+'/checkpoint'):
    os.mkdir(str(root_dir)+'/checkpoint')

print()
print("Epoch", "TR-loss", "TS-loss", sep=' '*8, end="\n")

for epoch in range(EPOCH):

    trainloss = train(epoch)
    testloss  = test(epoch)

    print(f"{epoch+0:03}/{EPOCH}", f"{trainloss:.4f}", f"{testloss:.4f}", sep=' '*8, end="\n")

    if SAVE:
        # Saving the model.
        is_best = testloss < BESTLOSS
        BESTLOSS = min(testloss, BESTLOSS)
        if is_best:
            torch.save(model.state_dict(), str(root_dir)+"/checkpoint/pacing_"+str(epoch)+".pt")
            print("Model Saved.")
print("="*50)
