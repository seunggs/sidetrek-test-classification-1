from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

@dataclass
class Hyperparameters:
    n_epochs: int = 3
    batch_size_train: int = 64
    batch_size_valid: int = 64
    batch_size_test: int = 1000
    lr: float = 0.01
    log_interval: int = 10
    random_seed: int = 42
    norm1: int = 0.1307
    norm2: int = 0.3081

hp = Hyperparameters()


class MNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.nll_loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.nll_loss(y_hat, y)
        self.log("test_loss", test_loss)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=hp.lr)

# Model
mnist = MNIST()

# Datasets
train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((hp.norm1,), (hp.norm2,))]),
)
test_dataset = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((hp.norm1,), (hp.norm2,))]),
)

train_dataset_size = int(len(train_dataset) * 0.8)
valid_dataset_size = len(train_dataset) - train_dataset_size

seed = torch.Generator().manual_seed(hp.random_seed)
train_dataset, valid_dataset = random_split(train_dataset, [train_dataset_size, valid_dataset_size], generator=seed)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=hp.batch_size_train, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=hp.batch_size_valid, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=hp.batch_size_test, shuffle=True)

# Trainer
trainer = pl.Trainer()

# Tasks
def train():
    trainer.fit(model=mnist, train_dataloaders=train_loader)
    torch.save()
    return

def test():
    trainer.test(model=mnist, dataloaders=test_loader)