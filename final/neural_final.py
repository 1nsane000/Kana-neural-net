import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as t
import torch.utils.data as data
import torch.optim as optim
import glob as g

import numpy as np

import matplotlib.pyplot as plt
import datetime

import time


device = torch.device("cuda")

train_path = 'datasets\\train_KATAKANA'
test_path = 'datasets\\test_KATAKANA'

#train_path = 'datasets\\train_HIRAGANA'
#test_path = 'datasets\\test_HIRAGANA'

num_epochs = 10
batch_size = 32
learning_rate = 0.0001
mean = 0.8964
std = 0.1064



def make_labels(path):
    file_list = g.glob(path)
    list = []
    for f in file_list:
        print(f)
        list.append(chr(int(f[f.rfind('\\')+1: ].upper(), base=16)))
    return list

def train_net(net_normal, train_loader, test_loader, optimizer, loss_fn, fw):
    val_losses = []
    train_losses = []
    for epoch in range(num_epochs):
    
        loss_avg = 0.0
        net_normal.train()
        for i, (batch) in enumerate(train_loader):
            inp, target = batch
            inp = inp.to(device)
            target = target.to(device)
            output = net_normal(inp)

            loss = loss_fn(output, target)
            loss_avg += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}]')
    
        valid_avg = 0.0
        net_normal.eval()
        for i, (batch) in enumerate(test_loader):
        
        
            inp, target = batch
            inp = inp.to(device)
            target = target.to(device)
            output = net_normal(inp)

            loss = loss_fn(output, target)
            valid_avg += loss.item()
        train_losses.append(loss_avg/len(train_loader))
        val_losses.append(valid_avg/len(test_loader))
        print(f'{net_normal.type} Epoch {epoch+1} \t\t Training Loss: {loss_avg / len(train_loader)} \t\t Validation Loss: {valid_avg / len(test_loader)}')
        fw.write(f'{net_normal.type} Epoch {epoch+1}'.ljust(26) + f'Training Loss: {loss_avg / len(train_loader)}'.ljust(42) + f'Validation Loss: {valid_avg / len(test_loader)}\n')


    model_file_name = f'{net_normal.type}_japanese_katakana-' + datetime.datetime.now().strftime("%I-%M-%p-%B-%d-%Y") + '.pt'
    torch.save(net_normal.state_dict(), ".\\params\\"+model_file_name)


def test_net(net_normal, test_loader, fw):
    acc = 0;
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for inp, labels in test_loader:
            inp = inp.to(device)
            labels = labels.to(device)
            outputs = net_normal(inp)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predicted == labels).sum().item()
        

        acc = 100.0 * n_correct / n_samples
        print(f'{net_normal.type} Accuracy: {acc}% ({n_correct}/{n_samples})')
        fw.write(f'{net_normal.type} Accuracy: {acc}% ({n_correct}/{n_samples})\n')

class NeuralNet(nn.Module):
    def __init__(self, c1=32, ln=1, t="normal"):
        super(NeuralNet, self).__init__()
        self.type = t
        self.c1 = c1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 86, kernel_size=4, padding=2)
        self.conv3 = nn.Conv2d(86, 129, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(129, 129, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(129, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(1024, 1024)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, ln)

        self.fc1n = nn.Linear(1024, 1024)
        self.fc2n = nn.Linear(1024, 512)
        self.fc3n = nn.Linear(512, ln)

    def forward(self, x):
        out = x
        out = self.maxpool(torch.relu(self.conv1(out)))
        out = self.maxpool(torch.relu(self.conv2(out)))

        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))
        out = torch.relu(self.conv5(out))
        out = self.maxpool(out)
        out = out.view(-1, 1024)

        if 'synergy' in self.type:
            out_normal = torch.relu(self.fc1(out))
            out_normal = self.drop(out_normal)
            out_normal = torch.relu(self.fc2(out_normal))
            out_normal = self.drop(out_normal)
            out_normal = self.fc3(out_normal)

            out_negative = torch.relu(self.fc1n(1-out))
            out_negative = self.drop(out_negative)
            out_negative = torch.relu(self.fc2n(out_negative))
            out_negative = self.drop(out_negative)
            out_negative = self.fc3n(out_negative)

            out = out_normal + out_negative
        else:
            if 'hybrid' in self.type:
                out = 1-out
            out = torch.relu(self.fc1(out))
            out = self.drop(out)
            out = torch.relu(self.fc2(out))
            out = self.drop(out)
            out = self.fc3(out)
        return out

transforms = t.Compose([t.Resize(size = (64,64)), 
                                 t.ToTensor(),
                                 t.Grayscale(),
                                 t.RandomPerspective(0.5,0.5),
                                 t.Normalize((mean),(std))])

transforms_test = t.Compose([t.Resize(size = (64,64)), 
                                 t.ToTensor(),
                                 t.Grayscale(),
                                 t.RandomPerspective(0.7,0.8),
                                 t.Normalize((mean),(std))])

train_data = torchvision.datasets.ImageFolder(root = train_path,
                                             transform = transforms)
test_data = torchvision.datasets.ImageFolder(root = test_path,
                                             transform = transforms_test)

classes = make_labels(train_path + '\\*')

num_classes = len(classes)
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle = True)

test_loader = data.DataLoader(test_data, batch_size=batch_size)

net_normal = NeuralNet(ln=num_classes).to(device)
conv1 = net_normal.conv1
conv2 = net_normal.conv2
conv2.weight.required_grad = False
conv1.weight.required_grad = False
net_synergy = NeuralNet(ln=num_classes, t='synergy').to(device)
net_hybrid = NeuralNet(ln=num_classes, t='hybrid').to(device)

loss_fn_n = nn.CrossEntropyLoss()
loss_fn_s = nn.CrossEntropyLoss()
loss_fn_h = nn.CrossEntropyLoss()

optimizer_n = optim.Adam(net_normal.parameters(), lr=learning_rate)
optimizer_s = optim.Adam(net_synergy.parameters(), lr=learning_rate)
optimizer_h = optim.Adam(net_hybrid.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)

val_losses = []
train_losses = []

log_name = 'log_' + datetime.datetime.now().strftime("%I-%M-%p-%B-%d-%Y") + '.txt'
fw = open('.\\logs\\' + log_name, 'w', encoding="utf-8")
fw.write(f'Epochs: {num_epochs}\n')
fw.write(f'Batch size: {batch_size}\n')
fw.write(f'Learning Rate: {learning_rate}\n\n')

start = time.time()

train_net(net_normal, train_loader, test_loader, optimizer_n, loss_fn_n, fw)
train_net(net_synergy, train_loader, test_loader, optimizer_s, loss_fn_s, fw)
train_net(net_hybrid, train_loader, test_loader, optimizer_h, loss_fn_h, fw)

fw.write('\n')

test_net(net_normal, test_loader, fw)
test_net(net_synergy, test_loader, fw)
test_net(net_hybrid, test_loader, fw)

end = time.time()

fw.write(f'Elapsed time: {end-start} seconds')
fw.close()

'''plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()'''
