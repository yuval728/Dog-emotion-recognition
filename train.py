import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import models
from dataset import get_datasets
from model import DogEmotionResNet
from utils import train, validate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'data/'

train_dataset, test_dataset, classes = get_datasets(data_dir)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

with open('classes.txt', 'w') as f:
    for item in classes:
        f.write("%s\n" % item)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

resnet_weights = models.ResNet50_Weights.DEFAULT
model = DogEmotionResNet(num_classes=len(classes), weights=resnet_weights)
model.to(device)

EPOCHS = 30
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in tqdm(range(EPOCHS)):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f'Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='train accuracy')
plt.plot(test_accuracies, label='test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'dog_emotion_model.pth')
