from __future__ import absolute_import, print_function

# --- System ---
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# --- Utility ---
import pandas as pd
import numpy as np
import math
import random
import logging
import pickle
from tqdm import tqdm
from datetime import datetime

# --- Pytorch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

# --- sklearn ---
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

# --- Plot ---
import matplotlib.pyplot as plt
import seaborn as sns

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
dataPath = "../data/statistics-5.csv"
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
# Dropping rows with pacing rate 10.5, glitch in the training data
df.drop( df[ df['PACING'] == 10.5 ].index, inplace=True)
num_of_classes = len(df['PACING'].unique())

# df['CONGESTION (Sender)'] = (df['CONGESTION (Sender)'] == 'cubic').astype(int)
# df['ALIAS'] = pd.factorize(df['ALIAS'])[0]

# Creating One-Hot Encoding for two columns.
mlb = MultiLabelBinarizer(sparse_output=True)
alias_df = df.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(df.pop('ALIAS')),
                                                            index=df.index,
                                                            columns=mlb.classes_))

df_ = alias_df.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(alias_df.pop('CONGESTION (Sender)')),
                                                            index=alias_df.index,
                                                            columns=mlb.classes_),
                                                            how = 'left', lsuffix='left', rsuffix='right')


X = df_[df_.columns.values].values
y = df_['PACING'].values
y = y.astype('float')

# Normalization
minmax_scale = preprocessing.MinMaxScaler().fit(df_[df_.columns.values])
df_minmax = minmax_scale.transform(df_[df_.columns.values])

final_df = pd.DataFrame(df_minmax, columns=df_.columns.values)
X = final_df[df_.columns.values].values

# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state=1)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test  = torch.tensor(X_test)
y_test  = torch.tensor(y_test)


# Hyperparameters
EPOCH = 300
BATCH = 256
LEARNING_RATE = 0.001

INTERVAL = 50
SAVE = True # False
BESTLOSS = 10

lossfn  = nn.CrossEntropyLoss()
# BCE = nn.BCELoss(reduction='mean')
# MSE = nn.MSELoss(reduction='mean') # 'mean', 'sum'. 'none'

# Custom data loader for ELK stack dataset
class PacingDataset(Dataset):
    """ TensorDataset with support of transforms. """
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
traindata   = PacingDataset(tensors=(X_train, y_train), transform=None)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=BATCH)
testdata    = PacingDataset(tensors=(X_test, y_test), transform=None)
testloader = torch.utils.data.DataLoader(testdata, batch_size=1)

inputFea = len(traindata[0][0])

# model definition
class PacingClassifier (nn.Module):
    # https://visualstudiomagazine.com/Articles/2021/02/11/pytorch-define.aspx?Page=2
    def __init__(self, nc=20, inputFeatures=7):
        super(PacingClassifier, self).__init__()

        self.fc1 = torch.nn.Linear(inputFeatures, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        self.fc3 = torch.nn.Linear(64, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, nc)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.zeros_(self.fc4.bias)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.zeros_(self.fc5.bias)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        torch.nn.init.zeros_(self.fc6.bias)

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x):
        z = self.lrelu(self.fc1(x))
        z = self.lrelu(self.fc2(z))
        z = self.lrelu(self.fc3(z))
        z = self.lrelu(self.fc4(z))
        z = self.lrelu(self.fc5(z))
        z = self.fc6(z)  # no activation
        return z

model = PacingClassifier (nc=num_of_classes, inputFeatures=inputFea)
print(model)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 350], gamma=0.1)

print("\nBatch Size = %3d " % BATCH)
print("Loss = " + str(lossfn))
print("Optimizer = SGD")
print("Max Epochs = %3d " % EPOCH)
print("Learning Rate = %0.3f " % LEARNING_RATE)
print("Number of Classes = %d " % num_of_classes)
print("\nStarting training ...")

model.train()
trainloss = []
for epoch in range(0, EPOCH):
    torch.manual_seed(epoch+1)              # recovery reproducibility
    epoch_loss = 0                          # for one full epoch

    for (batch_idx, batch) in enumerate(trainloader):
        (xs, ys) = batch                    # (predictors, targets)
        xs, ys = xs.float(), ys.float()
        optimizer.zero_grad()               # prepare gradients

        output = model(xs)                  # predicted pacing rate
        loss = lossfn(output, ys.long())    # avg per item in batch

        epoch_loss += loss.item()           # accumulate averages
        loss.backward()                     # compute gradients
        optimizer.step()                    # update weights
    
    scheduler.step()
    trainloss.append(epoch_loss)
    if epoch % INTERVAL == 0:

        model.eval()                        # evaluation phase
        correct, acc = 0, 0
        with torch.no_grad():
            for xs, ys in testloader:
                xs, ys = xs.float(), ys.long()
                pred = torch.max(model(xs), 1)[1]
                correct += (pred == ys).sum().item()
            acc = (100 * float(correct / len(testdata)) )

        print("Epoch = %4d      Loss = %0.4f      Accuracy = %0.4f" % (epoch, epoch_loss, acc))

        dt = time.strftime("%Y_%m_%d-%H_%M_%S")
        fn = "../checkpoint/" + str(dt) + str("-") + str(epoch) + "_ckpt.pt"

        info_dict = {
            'epoch' : epoch,
            'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict()
        }
        if SAVE:
            torch.save(info_dict, fn)       # save checkpoint

print("\nDone")