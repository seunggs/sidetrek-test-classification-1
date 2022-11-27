import typing
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from flytekit.types.file import FlyteFile
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

# Constants
MODEL_PATH = "model/mnist.pt"

# Metrics for logging
accuracy = Accuracy()


@dataclass_json
@dataclass
class Hyperparameters:
    max_epochs: int = 5
    batch_size_train: int = 64
    batch_size_valid: int = 64
    batch_size_test: int = 1000
    lr: float = 0.01
    log_interval: int = 10
    random_seed: int = 42
    norm1: int = 0.1307
    norm2: int = 0.3081


class MNIST(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
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

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("accuracy", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Datasets
def load_train_data(norm1, norm2, random_seed, batch_size_train, batch_size_valid):
    train_dataset = torchvision.datasets.MNIST(
        "files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((norm1,), (norm2,))]),
    )

    train_dataset_size = int(len(train_dataset) * 0.8)
    valid_dataset_size = len(train_dataset) - train_dataset_size

    seed = torch.Generator().manual_seed(random_seed)
    train_dataset, valid_dataset = random_split(train_dataset, [train_dataset_size, valid_dataset_size], generator=seed)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=False)

    return train_loader, valid_loader


def load_test_data(norm1, norm2, batch_size_test):
    test_dataset = torchvision.datasets.MNIST(
        "files/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((norm1,), (norm2,))]),
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return test_loader


# Tasks
TrainingOutputs = typing.NamedTuple(
    "TrainingOutputs",
    model_state=FlyteFile,
)


def train_mnist_task(hp: Hyperparameters) -> TrainingOutputs:
    train_loader, valid_loader = load_train_data(
        norm1=hp.norm1, norm2=hp.norm2, random_seed=hp.random_seed, batch_size_train=hp.batch_size_train, batch_size_valid=hp.batch_size_valid
    )
    model = MNIST(lr=hp.lr)
    trainer = pl.Trainer(log_every_n_steps=hp.log_interval, max_epochs=hp.max_epochs)
    trainer.fit(model, train_loader, valid_loader)

    torch.save(model.state_dict(), MODEL_PATH)

    return TrainingOutputs(model_state=FlyteFile(MODEL_PATH))


# def test_mnist_task(hp: Hyperparameters) -> TrainingOutputs:
#     test_loader = load_test_data(norm1=hp.norm1, norm2=hp.norm2, batch_size_test=hp.batch_size_test)
#     trainer = pl.Trainer(log_every_n_steps=hp.log_interval, max_epochs=hp.max_epochs)
#     trainer.test(dataloaders=test_loader)


def get_sample(hp: Hyperparameters):
    test_loader = load_test_data(norm1=hp.norm1, norm2=hp.norm2, batch_size_test=1)
    x, y = next(iter(test_loader))
    return x[0]


def predict(hp: Hyperparameters, input):
    model = MNIST(lr=hp.lr)
    model.load_state_dict(torch.load(MODEL_PATH))
    output = model(input)
    _, pred = torch.max(output, dim=1)
    return pred


# Test one sample and compare it with prediction from a saved model
def test_sample():
    input = get_sample(hp=Hyperparameters())
    plt.imshow(input[0], cmap="gray", interpolation="none")
    plt.show()
    pred = predict(hp=Hyperparameters(), input=input)
    print(pred)


if __name__ == "__main__":
    train_mnist_task(hp=Hyperparameters())
