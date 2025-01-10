import os
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

NUM_WORKERS = os.cpu_count()


def create_dataloaders(root_path, transform, batch_size, num_workers=NUM_WORKERS):
    # ImageFolder
    data = datasets.ImageFolder(root_path, transform=transform)

    # Split the data train-test: 80-20
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])

    # Get class names
    class_names = data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, class_names
