# Importing all dependencies
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt


# @dataclass
# class Hyperparameters(object):
#   """
#   n_epochs
#   batch_size_train
#   batch_size_test
#   learning_rate
#   momentum
#   log_interval
#   random_seed
#   """
#   n_epochs = 3
#   batch_size_train = 64
#   batch_size_test = 1000
#   learning_rate = 0.01
#   momentum = 0.5
#   log_interval = 10
#   random_seed = 42
#   norm1 = 0.1307
#   norm2 = 0.3081

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 42
norm1 = 0.1307
norm2 = 0.3081

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# Loading the datasets
def load_dataset(norm1, norm2, batch_size_train, batch_size_test):
  # Loading train dataset
  train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (norm1,), (norm2,))
                             ])), batch_size=batch_size_train, shuffle=True)
  # Loading test dataset
  test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (norm1,), (norm2,))
                             ])), batch_size=batch_size_test, shuffle=True)
  return train_loader, test_loader

train_loader, test_loader = load_dataset(norm1, norm2, batch_size_train, batch_size_test)

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)


# Building network
def build_network():
  class Net(nn.Module):
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
  return Net()

network = build_network()

# Creating the optimizer
def create_optimizer(learning_rate, momentum):
  return optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)

optimizer = create_optimizer(learning_rate, momentum)

# Training model on train data
def train_model(n_epochs, network, train_loader, optimizer, log_interval):
  train_losses = []
  train_counter = []
  # Training model
  def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = network(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
          (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    return train_counter, train_losses
  return train, network, train_counter, train_losses

train, network, train_counter, train_losses = train_model(n_epochs, network, train_loader, optimizer, log_interval)

# Testing model on test data
def test_model(n_epochs, network, train_loader, test_loader):
  test_losses = []
  test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

  def test():
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
    result = ('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
    print(result)
    return result, test_losses
  result, test_losses = test()
  # test()
  for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
  return result, test_counter, test_losses

result, test_counter, test_losses = test_model(n_epochs, network, train_loader, test_loader)

# Plotting and saving the loss
def save_loss(train_counter, train_losses, test_counter, test_losses):
  fig = plt.figure(figsize=(8,5))
  plt.plot(train_counter, train_losses, color='blue')
  plt.scatter(test_counter, test_losses, color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('number of training examples seen')
  plt.ylabel('negative log likelihood loss')
  plt.savefig('loss.png')
  print("Loss curve is saved as loss.png")
  return "Loss curve is saved as loss.png"

save_loss(train_counter, train_losses, test_counter, test_losses)


# Predicting digits
def predict_digits(test_loader, network):
  examples = enumerate(test_loader)
  batch_idx, (example_data, example_targets) = next(examples)
  with torch.no_grad():
    output = network(example_data)
  fig = plt.figure()
  for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
      output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
  plt.savefig('prediction.png')
  print("Prediction for 9 digits is saved as prediction.png")
  return "Prediction for 9 digits is saved as prediction.png"

predict_digits(test_loader, network)

# Saving model
def save_model(network):
  torch.save(network.state_dict(), 'model.pth')
  print("Model is saved as model.pth")
  return "Model is saved as model.pth"

save_model(network)

# Saving optimizer
def save_optimizer(optimizer):
  torch.save(optimizer.state_dict(), 'optimizer.pth')
  print("Optimizer is saved as optimizer.pth")
  return "Optimizer is saved as optimizer.pth"

save_optimizer(optimizer)