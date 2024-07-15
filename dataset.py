import torch
from torchvision import datasets, transforms, models

def get_datasets(data_dir):
    resnet_weights = models.ResNet50_Weights.DEFAULT
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    image_dataset = datasets.ImageFolder(data_dir, transform=resnet_weights.transforms())
    
    train_size = int(0.9 * len(image_dataset))
    test_size = len(image_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    return train_dataset, test_dataset, image_dataset.classes
