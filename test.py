import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import get_datasets
from model import DogEmotionResNet
from utils import predict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = 'data/'

_, test_dataset, classes = get_datasets(data_dir)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = DogEmotionResNet(num_classes=len(classes))
model.load_state_dict(torch.load('dog_emotion_model.pth'))
model.to(device)

predictions = predict(model, test_loader, device)

print(classification_report([label for _, label in test_dataset], predictions))
