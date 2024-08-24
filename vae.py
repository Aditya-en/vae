import torch
import torch.nn as nn
import os
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
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
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

def loss_function(recon_x, x, mu, logvar, beta=0.2):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

model = VAE(input_channels=3, hidden_dim=64, latent_dim=128, img_size=256).to(device)
model.load_state_dict(torch.load("vae.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create directory to save model and samples if they don't exist
os.makedirs('samples', exist_ok=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # Print current batch index
        print(f'Batch {batch_idx+1}/{len(dataloader)} || loss {loss.item():.5f}')

    # Print epoch loss
    print(f'Epoch {epoch+1}, Loss: {train_loss/len(dataloader.dataset):.4f}')

    # Save model
    model_path = f'vae.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # Generate samples
    model.eval()
    with torch.no_grad():
        sample = torch.randn(16, model.latent_dim).to(device)  # Assuming latent_dim is defined in the model
        sample = model.decode(sample).cpu()
        sample_path = f'samples/sample_modifiedLoss_epoch_{epoch+1}.png'
        vutils.save_image(sample, sample_path, nrow=4)
        print(f'Samples saved to {sample_path}')


# model.load_state_dict(torch.load("models/vae_epoch_1.pth"))
# model.eval()
# with torch.no_grad():
#     sample = torch.randn(16, model.latent_dim).to(device)  # Assuming latent_dim is defined in the model
#     sample = model.decode(sample).cpu()
#     sample_path = f'samples/sample_epoch1.png'
#     vutils.save_image(sample, sample_path, nrow=4)
#     print(f'Samples saved to {sample_path}')
