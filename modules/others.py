# Visualize a image
def plot_an_image(dataloader, class_names):
    # Get a batch of images
    image_batch, label_batch = next(iter(dataloader))

    # Get a single image from the batch
    image, label = image_batch[0], label_batch[0]

    # View the batch shapes
    print(image.shape, label)

    # Plot image with matplotlib
    # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.imshow(image.permute(1, 2, 0))
    plt.title(class_names[label])
    plt.axis(False)


# Set seeds
def set_seeds(seed=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
