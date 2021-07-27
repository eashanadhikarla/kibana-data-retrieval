import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model definition
class PacingClassifier (nn.Module):
    # https://visualstudiomagazine.com/Articles/2021/02/11/pytorch-define.aspx?Page=2
    def __init__(self, nc=21, inputFeatures=7):
        super(PacingClassifier, self).__init__()

        self.fc1 = torch.nn.Linear(inputFeatures, 32)
        self.drop1 = torch.nn.Dropout(0.25)
        # ----------------------------------
        self.fc2 = torch.nn.Linear(32, 64)
        self.drop2 = torch.nn.Dropout(0.70)
        # ----------------------------------
        self.fc3 = torch.nn.Linear(64, 64)
        self.drop3 = torch.nn.Dropout(0.70)
        # ----------------------------------
        self.fc4 = torch.nn.Linear(64, 32)
        self.drop4 = torch.nn.Dropout(0.70)
        # ----------------------------------
        self.fc5 = torch.nn.Linear(32, nc)

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

    def forward(self, x):
        z = self.fc1(x)
        z = torch.relu(z)
        z = self.drop1(z)

        z = self.fc2(z)
        z = torch.relu(z)
        z = self.drop2(z)

        z = self.fc3(z)
        z = torch.relu(z)
        z = self.drop3(z)

        z = self.fc4(z)
        z = torch.relu(z)
        z = self.drop4(z)

        z = self.fc5(z)  # no activation
        return z