import torch
import torch.nn as nn
from torchsummary import summary

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
model.load_state_dict(torch.load("vae.pth"))
model.to(device)  # Move the model to the appropriate device

# Print the summary, including the input size, and ensure the input tensor is on the same device
summary(model, input_size=(3, 256, 256), device=device.type)
