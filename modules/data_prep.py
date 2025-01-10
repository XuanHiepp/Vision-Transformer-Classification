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


def process_data(image, patch_size):
    # 1. Shape of image tensor
    height, width = image.shape[1], image.shape[2]

    # 2. Get image tensor
    x = image.unsqueeze(0)

    # 3. Create patch embedding layer
    patch_embedding_layer = PatchEmbedding(in_channels=3,
                                           patch_size=patch_size,
                                           embedding_dim=768)

    # 4. Pass image through patch embedding layer
    patch_embedding = patch_embedding_layer(x)

    # 5. Create class token embedding
    batch_size = patch_embedding.shape[0]
    embedding_dimension = patch_embedding.shape[-1]
    class_token = nn.Parameter(torch.ones(
        batch_size, 1, embedding_dimension), requires_grad=True)

    # 6. Prepend class token embedding to patch embedding
    patch_embedding_class_token = torch.cat(
        (class_token, patch_embedding), dim=1)

    # 7. Create position embedding
    number_of_patches = int((height * width) / patch_size**2)
    position_embedding = nn.Parameter(torch.ones(
        1, number_of_patches+1, embedding_dimension), requires_grad=True)

    # 8. Add position embedding to patch embedding with class token
    patch_and_position_embedding = patch_embedding_class_token + position_embedding

    return patch_and_position_embedding
