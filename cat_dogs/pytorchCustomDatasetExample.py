import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm 
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from customDataset import CateAndDogsDataset
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

in_channels = 3
num_class =  10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

dataset = CateAndDogsDataset(csv_file='cat_dogs\cats_dogs.csv', root_dir="cat_dogs\cats_dogs_resized", transform = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [8, 2])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


model = torchvision.models.googlenet(pretrained=True)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in tqdm(range(num_epochs)):
    losses = []

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        scores = model(data)
        loss = criterion(scores, target)

        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")



def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()

print("Checking accuracy on trainning set")
check_accuracy(train_loader, model)

print("Checking accuracy on testing set")
check_accuracy(test_loader, model)



