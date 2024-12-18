import torch
import torch.nn as nn
from torchsummary import summary
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.utils as vutils

class VAE(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim, img_size):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the flattened features
        self.flatten_size = hidden_dim * 8 * (img_size // 16) ** 2

        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.hidden_dim*8, self.img_size//16, self.img_size//16)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Determine the device to use
device = torch.device("cpu")

# Instantiate and load the model
model = VAE(input_channels=3, hidden_dim=64, latent_dim=128, img_size=256)
model.load_state_dict(torch.load("vae.pth", weights_only=True))
model.to(device)  # Move the model to the appropriate device

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
class LandscapeDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def collect_image_paths(root_dir):
    image_paths = []
    for split in ["Training Data", "Validation Data", "Testing Data"]:
        split_dir = os.path.join(root_dir, split)
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
    return image_paths

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Collect image paths
root_dir = "Landscape Classification"
image_paths = collect_image_paths(root_dir)

# Create a single dataset and dataloader for training
dataset = LandscapeDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

images = next(iter(dataloader))
reconstructed_images, _, _ = model(images)
vutils.save_image(images, "test_images.png", nrow=4)
vutils.save_image(reconstructed_images, "reconstructed_images.png", nrow=4)

# image = Image.open(r"Landscape Classification\Validation Data\Glacier\Glacier-Valid (2).jpeg")
# print(type(image))
# image = transform(image)
# print(type(image))
# reconstructed_image, _, _ = model(image.unsqueeze(0))
# print(reconstructed_image.shape)

# Print the summary, including the input size, and ensure the input tensor is on the same device
# summary(model, input_size=(3, 256, 256), device=device.type)
