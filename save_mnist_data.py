
import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Directories
train_dir = "dataset/mnist"
val_dir = "dataset/mnist_val"

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Download the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)

# Calculate split sizes
total_size = len(mnist_dataset)
val_size = int(0.01 * total_size)  # 1% for validation
train_size = total_size - val_size  # Remaining 99% for training

# Split dataset
train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

# Save train dataset
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
for i, (image, label) in enumerate(train_loader):
    image_path = os.path.join(train_dir, f"{str(i).zfill(5)}.png")  # Sequentially numbered
    transforms.ToPILImage()(image.squeeze()).save(image_path)

# Save validation dataset
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
for i, (image, label) in enumerate(val_loader):
    image_path = os.path.join(val_dir, f"{str(i).zfill(5)}.png")  # Sequentially numbered
    transforms.ToPILImage()(image.squeeze()).save(image_path)

print(f"Training data saved in {train_dir}")
print(f"Validation data saved in {val_dir}")
