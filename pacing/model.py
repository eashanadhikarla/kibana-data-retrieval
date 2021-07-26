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

# -----------------------------------------------------------
# random weight initialization
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

# -----------------------------------------------------------
# data loading and preprocessing
dataPath = "data/statistics (pacing).csv"
df = pd.read_csv(dataPath)

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
X = final_df[['THROUGHPUT (Sender)', 'LATENCY (mean)', 'RETRANSMITS', 'STREAMS', 'CONGESTION (Sender)']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state=1)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test  = torch.tensor(X_test)
y_test  = torch.tensor(y_test) 

# -----------------------------------------------------------

# Custom data loader for ELK stack dataset
class PacingDataset(Dataset):
    """
    TensorDataset with support of transforms.
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

# -----------------------------------------------------------

# accuracy computation
def accuracy(model, ds, pct):
    # assumes model.eval()
    # percent correct within pct of true pacing rate
    n_correct = 0; n_wrong = 0

    for i in range(len(ds)):
        (X, Y) = ds[i]                # (predictors, target)
        X, Y = X.float(), Y.float()
        with torch.no_grad():
            output, _, _, _ = model(X)         # computed price

        abs_delta = np.abs(output.item() - Y.item())
        max_allow = np.abs(pct * Y.item())
        if abs_delta < max_allow:
            n_correct +=1
        else:
            n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc*100

# -----------------------------------------------------------

input_feature = 5
latent_feature = 16

class VAERegressor(nn.Module):
    def __init__(self):
        super(VAERegressor, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=input_feature, out_features=128)
        self.enc2 = nn.Linear(in_features=128, out_features=latent_feature*2)
        # decoder
        self.dec1 = nn.Linear(in_features=latent_feature, out_features=128)
        self.dec2 = nn.Linear(in_features=128, out_features=5)
        # Regressor
        self.fc1 = torch.nn.Linear (5, 32)
        self.fc2 = torch.nn.Linear (32, 1)

        torch.nn.init.xavier_uniform_(self.enc1.weight)
        torch.nn.init.zeros_(self.enc1.bias)
        torch.nn.init.xavier_uniform_(self.enc2.weight)
        torch.nn.init.zeros_(self.enc2.bias)
        torch.nn.init.xavier_uniform_(self.dec1.weight)
        torch.nn.init.zeros_(self.dec1.bias)
        torch.nn.init.xavier_uniform_(self.dec2.weight)
        torch.nn.init.zeros_(self.dec2.bias)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        
        # encoding
        x = self.enc1(x)
        x = F.relu(x)
        x = self.enc2(x).view(-1, 2, latent_feature)

        # get `mu` and `log_var`
        mu      = x[:, 0, :]    # the first feature values as mean
        log_var = x[:, 1, :]    # the other feature values as variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = self.dec1(z)
        x = F.relu(x)
        x = self.dec2(x)
        recon = torch.sigmoid(x)

        # regressor
        x = self.fc1(recon)
        x = F.relu(x)
        x = self.fc2(x)

        return x, recon, mu, log_var


# model definition
class PacingOptimizer(nn.Module):
    # https://visualstudiomagazine.com/Articles/2021/02/11/pytorch-define.aspx?Page=2
    def __init__(self):
        super(PacingOptimizer, self).__init__()
        self.hid1 = torch.nn.Linear(5, 32)
        self.drop1 = torch.nn.Dropout(0.25)
        self.hid2 = torch.nn.Linear(32, 64)
        self.drop2 = torch.nn.Dropout(0.50)
        self.hid3 = torch.nn.Linear(64, 32)
        self.oupt = torch.nn.Linear(32, 1)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.hid3.weight)
        torch.nn.init.zeros_(self.hid3.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = self.drop1(torch.relu(self.hid1(x)))
        z = self.drop2(torch.relu(self.hid2(z)))
        z = torch.relu(self.hid3(z))
        z = self.oupt(z)  # no activation
        return z

# -----------------------------------------------------------
model = VAERegressor()
# model = PacingOptimizer()

# Hyperparameters
EPOCH = 100
BATCH = 64
LEARNING_RATE = 0.005

INTERVAL = 50
SAVE = False
BESTLOSS = 10

CE  = nn.CrossEntropyLoss()
BCE = nn.BCELoss(reduction='mean')
MSE = nn.MSELoss(reduction='mean') # 'mean', 'sum'. 'none'

def criterion(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

print("\nBatch Size = %3d " % BATCH)
print("Loss = " + str(criterion))
print("Pptimizer = Adam")
print("Max Epochs = %3d " % EPOCH)
print("Learning Rate = %0.3f " % LEARNING_RATE)

# Dataset w/o any tranformations
traindata   = PacingDataset(tensors=(X_train, y_train), transform=None)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=BATCH)

testdata    = PacingDataset(tensors=(X_test, y_test), transform=None)
testloader = torch.utils.data.DataLoader(testdata, batch_size=BATCH)

print("\nStarting training with saved checkpoints")

model.train()
for epoch in range(0, EPOCH):
    torch.manual_seed(epoch+1) # recovery reproducibility
    epoch_loss = 0             # for one full epoch

    for (batch_idx, batch) in enumerate(trainloader):
        (xs, ys) = batch                # (predictors, targets)
        xs, ys = xs.float(), ys.float()
        optimizer.zero_grad()           # prepare gradients

        # output = model(xs)            # predicted pacing rate
        # loss = criterion(ys, output)  # avg per item in batch
        output, recon, mu, log_var = model(xs)
        mse_loss = MSE(ys, output)
        bce_loss = BCE(recon, xs)
        loss = criterion(bce_loss, mu, log_var) + mse_loss

        epoch_loss += loss.item()       # accumulate averages
        loss.backward()                 # compute gradients
        optimizer.step()                # update weights

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

# evaluate model accuracy
model.eval()

gap = 0.50
acc_train = accuracy(model, traindata, gap)
print(f"Accuracy (within {gap:.2f}) on train data = {acc_train:.2f}%")


# make prediction on a random sample from test data
pivot = random.randint(0, len(testdata))
print("Pivot: ", pivot)
x, y = testdata[pivot]
tput, lat, loss, streams, cong = x.detach().numpy()
# tput, lat, loss, streams, cong = 0.149677, 0.577766, 1.00000, 0.0, 1.0
print(f"\nPredicting pacing rate for:\n\
    (norm. values)\n\
    throughput = {tput}\n\
    latency = {lat}\n\
    loss = {loss}\n\
    congestion = {cong}\n\
    streams = {streams}\n\
    pacing = {y}")

# converting the sample to tensor array
ukn = np.array([[tput, lat, loss, streams, cong]], dtype=np.float32)
sample = torch.tensor(ukn, dtype=torch.float32).to(device)

# testing the sample
with torch.no_grad():
    pred, _, _, _ = model(sample)
pred = pred.item()
print(f"\nPredicted Pacing rate: {pred:.4f}\nGround-truth Pacing rate: {y:.4f}\n")
