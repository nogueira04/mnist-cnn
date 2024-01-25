import torch
import torch.optim as optim
import torch.nn as nn
from network1 import CNN
import matplotlib.pyplot as plt
from data import loaders, training_data, test_data

device = "cpu"
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

accuracies = []

def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch: {epoch} [{batch_idx * len(data)}]/{len(loaders['train'].dataset)} \
                    ({100. * batch_idx / len(loaders['train']):.0f}%]\t{loss.item():.6f})")


def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders["test"].dataset)
    accuracy = 100. * correct / len(loaders['test'].dataset)
    accuracies.append(accuracy)

    print(f"\nTest set: Avg loss: {test_loss:.4f}, \
           Accuracy: {correct}/{len(loaders['test'].dataset)} \
           ({accuracy:.0f})%\n")
    

def plot(epochs, scores):
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.plot(epochs)
    plt.plot(scores)
    plt.savefig("results.png")

for epoch in range(1, 31):
    train(epoch)
    test()

plot(30, accuracies)