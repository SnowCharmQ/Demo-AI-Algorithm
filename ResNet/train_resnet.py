from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ResNet18

BATCH_SIZE = 512
learning_rate = 1e-3
epochs = 300

cifar_train = datasets.CIFAR10('cifar', train=True, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomRotation([90, 180]),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]), download=True)
cifar_train = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True)

cifar_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]), download=True)
cifar_test = DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=True)

x, label = iter(cifar_train).__next__()

cuda_id = 0
device_name = "cuda:{}".format(cuda_id)
device = torch.device(device_name)
device_ids = [cuda_id]
net = ResNet18().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):

    net.train()
    loss = 0
    for batch_idx, (x, label) in enumerate(cifar_train):
        x, label = x.to(device), label.to(device)
        logits = net(x)

        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch: ', epoch, ', loss: ', loss.item())

    net.eval()
    with torch.no_grad():

        total_correct = 0
        total_num = 0

        for x, label in cifar_test:
            x, label = x.to(device), label.to(device)

            logits = net(x)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        print('Epoch: ', epoch, ', accuracy: ', acc)
