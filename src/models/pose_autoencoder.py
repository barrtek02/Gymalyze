from torch import nn


class PoseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(PoseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        x = x.view(batch_size * seq_length, -1)  # Flatten for fully connected layers
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(batch_size, seq_length, input_dim)  # Reshape back
        return decoded
