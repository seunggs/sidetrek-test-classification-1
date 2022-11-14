### Importing all dependencies
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Callable, Any, Dict, Tuple, List
import typing
from flytekit.types.file import PythonPickledFile
from flytekit.types.file import FlyteFile


@dataclass
class Hyperparameters(object):
    """
    n_epochs
    batch_size_train
    batch_size_test
    learning_rate
    momentum
    log_interval
    random_seed
    norm1
    norm2
    """

    n_epochs: int = 3
    batch_size_train: int = 64
    batch_size_test: int = 1000
    learning_rate: float = 0.01
    momentum: float = 0.5
    log_interval: int = 10
    random_seed: int = 42
    norm1: int = 0.1307
    norm2: int = 0.3081


### Instantiating the Hyperparameters class
hp = Hyperparameters()

### Creating the network
class Net(nn.Module):
    """
    This class returns the network
    """

    def __init__(self):
        super(Net, self).__init__()
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


### Instantiating the network
network = Net()

### Loading the datasets
def load_dataset(
    norm1: float, norm2: float, batch_size_train: int, batch_size_test: int, random_seed: int
) -> Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader]:
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    return (
        torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "files/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((norm1,), (norm2,))]),
            ),
            batch_size=batch_size_train,
            shuffle=True,
        ),
        torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                "files/",
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((norm1,), (norm2,))]),
            ),
            batch_size=batch_size_test,
            shuffle=True,
        ),
    )


### Creating the optimizer
def create_optimizer(network: PythonPickledFile, learning_rate: float, momentum: float) -> Dict:
    return optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


train_losses = []
train_counter = []
test_counter = []
test_losses = []

### Training model
def train(n_epochs: int, network: PythonPickledFile, train_loader: torch.utils.data.dataloader.DataLoader, optimizer: Dict) -> Tuple[List[int], List[float]]:
    train_losses = []
    train_counter = []
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % hp.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    n_epochs, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()
                )
            )
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((n_epochs - 1) * len(train_loader.dataset)))
    return train_counter, train_losses


### Creting test() function
def test(
    n_epochs: int, network: PythonPickledFile, test_loader: torch.utils.data.dataloader.DataLoader, train_loader: torch.utils.data.dataloader.DataLoader
) -> Tuple[str, List[float], List[int]]:
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    result = "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset))
    print(result)
    return result, test_losses, test_counter


### Testing model on test data
def create_training_loop(
    n_epochs: int,
    network: PythonPickledFile,
    optimizer: Dict,
    train_losses: List[float],
    train_counter: List[int],
    train_loader: torch.utils.data.dataloader.DataLoader,
    test_loader: torch.utils.data.dataloader.DataLoader,
) -> Tuple[List[int], List[float], List[int], List[float]]:

    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, optimizer)
        test(n_epochs, network, test_loader, train_loader)
    return test_counter, test_losses, train_counter, train_losses


### Plotting and saving the loss
def save_loss(train_counter: List[int], train_losses: List[float], test_counter: List[int], test_losses: List[float]) -> FlyteFile[typing.TypeVar("png")]:
    plt.figure(figsize=(8, 5))
    plt.plot(train_counter, train_losses, color="blue")
    plt.plot(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("Number of training examples seen")
    plt.ylabel("Negative log likelihood loss")
    print("Loss curve is saved as loss.png")
    return plt.savefig("loss2.png")


### Predicting digits
def predict_digits(test_loader: torch.utils.data.dataloader.DataLoader, network: PythonPickledFile) -> FlyteFile[typing.TypeVar("png")]:
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = network(example_data)
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
        plt.savefig("prediction.png")
    print("Prediction for 9 digits is saved as prediction.png")
    return plt.savefig("prediction.png")


### Saving model
def save_model(network: PythonPickledFile) -> FlyteFile[typing.TypeVar("pth")]:
    print("Model is saved as model.pth")
    return torch.save(network.state_dict(), "model.pth")


### Saving optimizer
def save_optimizer(optimizer: Dict) -> FlyteFile[typing.TypeVar("pth")]:
    print("Optimizer is saved as optimizer.pth")
    return torch.save(optimizer.state_dict(), "optimizer.pth")


### Creating Workflow
def create_workflow(
    network: PythonPickledFile, learning_rate: float, momentum: float, norm1: float, norm2: float, batch_size_train: int, batch_size_test: int, random_seed: int, n_epochs: int
) -> Tuple[FlyteFile[typing.TypeVar("png")], FlyteFile[typing.TypeVar("png")], FlyteFile[typing.TypeVar("pth")], FlyteFile[typing.TypeVar("pth")]]:
    optimizer = create_optimizer(network, learning_rate, momentum)
    train_loader, test_loader = load_dataset(norm1, norm2, batch_size_train, batch_size_test, random_seed)
    create_training_loop(n_epochs, network, optimizer, train_losses, train_counter, train_loader, test_loader)
    return (save_loss(train_counter, train_losses, test_counter, test_losses), predict_digits(test_loader, network), save_model(network), save_optimizer(optimizer))


if __name__ == "__main__":
    create_workflow(
        network=network,
        learning_rate=hp.learning_rate,
        momentum=hp.momentum,
        norm1=hp.norm1,
        norm2=hp.norm2,
        batch_size_train=hp.batch_size_train,
        batch_size_test=hp.batch_size_test,
        random_seed=hp.random_seed,
        n_epochs=hp.n_epochs,
    )
