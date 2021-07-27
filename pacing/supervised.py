from __future__ import absolute_import, print_function

# --- System ---
import os
import sys
import time
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

# --- Plot ---
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

from lib.dataloader import PacingDataset
from lib.classifier import PacingClassifier, resnet50
import lib.utils

# random weight initialization
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = os.getcwd()

# data loading and preprocessing
dataPath = "data/statistics-5.csv"
df = pd.read_csv(dataPath)
# ----------------------------------
# Dropping columns that are not required at the moment
df = df.drop(columns=['Unnamed: 0', 'UUID', 'HOSTNAME', 'TIMESTAMP', 'THROUGHPUT (Receiver)', 'LATENCY (mean)', 'CONGESTION (Receiver)', 'BYTES (Receiver)'])

# Pre-processing
pacing = df['PACING'].values
for i, p in enumerate(pacing):
    v, _ = p.split("gbit")
    pacing[i] = float(v) # int(v)

df['PACING'] = pacing
df['CONGESTION (Sender)'] = (df['CONGESTION (Sender)'] == 'cubic').astype(int)
df['ALIAS'] = pd.factorize(df['ALIAS'])[0]

num_of_classes = len(df['PACING'].unique())

X = df[['THROUGHPUT (Sender)', 'LATENCY (min.)', 'LATENCY (max.)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)', 'ALIAS']].values
y = df['PACING'].values
y = y.astype('int')

# Normalization
minmax_scale = preprocessing.MinMaxScaler().fit(df[['THROUGHPUT (Sender)', 'LATENCY (min.)', 'LATENCY (max.)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)', 'ALIAS']])
df_minmax = minmax_scale.transform(df[['THROUGHPUT (Sender)', 'LATENCY (min.)', 'LATENCY (max.)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)', 'ALIAS']])

final_df = pd.DataFrame(df_minmax, columns=['THROUGHPUT (Sender)', 'LATENCY (min.)', 'LATENCY (max.)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)', 'ALIAS'])
X = final_df[['THROUGHPUT (Sender)', 'LATENCY (min.)', 'LATENCY (max.)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)', 'ALIAS']].values
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state=1)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test  = torch.tensor(X_test)
y_test  = torch.tensor(y_test)

# Hyperparameters
EPOCH = 1000
BATCH = 512
LEARNING_RATE = 0.01

INTERVAL = 50
SAVE = False
BESTLOSS = 10

CE  = nn.CrossEntropyLoss()
BCE = nn.BCELoss(reduction='mean')
MSE = nn.MSELoss(reduction='mean') # 'mean', 'sum'. 'none'

# Dataset w/o any tranformations
traindata   = PacingDataset(tensors=(X_train, y_train), transform=None)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=BATCH)

testdata    = PacingDataset(tensors=(X_test, y_test), transform=None)
testloader = torch.utils.data.DataLoader(testdata, batch_size=1) # BATCH)

inputFea = len(traindata[0][0])
model = PacingClassifier (nc=num_of_classes, inputFeatures=inputFea)
# model = resnet50(device=device)
print(model)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[350,500], gamma=0.05)

print("\nBatch Size = %3d " % BATCH)
print("Loss = " + str(CE))
print("Optimizer = SGD")
print("Max Epochs = %3d " % EPOCH)
print("Learning Rate = %0.3f " % LEARNING_RATE)
print("Number of Classes = %d " % num_of_classes)

print("\nStarting training with saved checkpoints")

model.train()
for epoch in range(0, EPOCH):
    torch.manual_seed(epoch+1) # recovery reproducibility
    epoch_loss = 0             # for one full epoch

    for (batch_idx, batch) in enumerate(trainloader):
        (xs, ys) = batch                # (predictors, targets)
        xs, ys = xs.float(), ys.float()
        optimizer.zero_grad()           # prepare gradients

        output = model(xs)              # predicted pacing rate
        loss = CE(output, ys.long())    # avg per item in batch

        epoch_loss += loss.item()       # accumulate averages
        loss.backward()                 # compute gradients
        optimizer.step()                # update weights
    
    scheduler.step()
    if epoch % INTERVAL == 0:
        print("Epoch = %4d    Loss = %0.4f" % (epoch, epoch_loss))

        # save checkpoint
        dt = time.strftime("%Y_%m_%d-%H_%M_%S")
        fn = str(dt) + str("-") + str(epoch) + "_ckpt.pt"

        info_dict = {
            'epoch' : epoch,
            'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict()
        }
        if SAVE:
            torch.save(info_dict, fn)

print("\nDone")


correct, acc, total = 0, 0, 0
running_loss = 0
with torch.no_grad():
    for xs, ys in testloader:
        xs, ys = xs.float(), ys.long()

        output = model(xs)
        loss = CE(output, ys.long())

        running_loss += loss.item()
        total += ys.size(0)
        pred = torch.max(output, 1)[1]
        correct += (pred == ys).sum().item()
    acc = (100 * float(correct / total) )
print(f"Accuracy: {acc:.3f}%")
