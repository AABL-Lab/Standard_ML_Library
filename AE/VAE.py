"""
This code mainly follows the Geeks4Geeks Pytorch Autoencoder example.
https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/

Any modifiations are made by the AABL Lab.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim=10, learning_rate=1e-3):
        super().__init__()
        #TODO: It might be good to be able to calculate reduction till it hits desired latent space
        # Perhaps in a loop, etc. but for the sake of initial implementation I have not done this.
        self.input_dim = input_dim # Input dimensions of data

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim//2)),
            nn.ReLU(),
            nn.Linear((input_dim//2), (input_dim//4)),
            nn.ReLU(),
            nn.Linear((input_dim//4), (input_dim//8)),
            nn.ReLU(),
            nn.Linear((input_dim//8), (input_dim//16)),
            nn.ReLU(),
            nn.Linear((input_dim//16), (input_dim//32)),
            nn.ReLU(),
            nn.Linear((input_dim//32), (input_dim//64))
        )

        self.mu = nn.Linear((input_dim//64), latent_dim)
        self.var = nn.Linear((input_dim//64), latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (input_dim//64)),
            nn.Linear((input_dim//64), (input_dim//32)),
            nn.ReLU(),
            nn.Linear((input_dim//32), (input_dim//16)),
            nn.ReLU(),
            nn.Linear((input_dim//16), (input_dim//8)),
            nn.ReLU(),
            nn.Linear((input_dim//8), (input_dim//4)),
            nn.ReLU(),
            nn.Linear((input_dim//4),  (input_dim//2)),
            nn.ReLU(),
            nn.Linear((input_dim//2), input_dim),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        encoded = self.encoder(x)

        mu = self.mu(encoded)
        var = self.var(encoded)
        z = self.reparameterize(mu, var)

        decoded = self.decoder(z)
        return decoded,  mu, var

    def train(self, data_loader, epochs, print_epoch_loss=True):
        for epoch in range(epochs):
            loss = 0
            for batch_data, _ in data_loader:
                # Reshape mini-batch data to [N, 784] matrix
                batch_data = batch_data.view(-1, self.input_dim)
                
                self.optimizer.zero_grad() # Zero gradient for training
                
                outputs, mu, var = self.forward(batch_data) # Feed forwards and get outputs
                
                reconstruction_loss = self.criterion(outputs, batch_data) # Compute loss
                kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)

                kld_weight = 0.1 # NOTE: I know there are ways to calulate this... I just haven't. Forgive me.
                #TODO: Actually get kld weight the correct way
                train_loss = reconstruction_loss + kld_weight * kld_loss
                train_loss.backward() # Compute backprop gradients from loss
                
                self.optimizer.step() # Update with computed gradients
                
                # Add the mini-batch training loss to epoch loss
                loss += train_loss.item()
            
            # Compute the epoch training loss
            loss = loss / len(train_loader)
            
            # Display the epoch training loss
            if print_epoch_loss:
                print("Epoch : {}/{}, Loss = {:.6f}".format(epoch + 1, epochs, loss))
    

if __name__ == "__main__":
    model = VAE(input_dim=784) # Dimensions of MNIST images flattened (28 x 28 = 784)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    model.train(train_loader, 100)
